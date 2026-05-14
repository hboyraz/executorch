# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from executorch.backends.transforms.permute_pass_utils import (
    get_arg,
    get_permuted_dims,
    get_transposed_dims,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch.fx import Node


class FuseCascadedTransposeOrPermuteOps(RemoveOrReplacePassInterface):
    """
    Fuse a chain of transpose and permute ops into a single permute or a no-op.
    Handles branches and chains of permutes, including permute-view-permute
    patterns where a squeeze/unsqueeze view sits between two permutes.
    """

    transpose_or_permute_target = {
        exir_ops.edge.aten.transpose_copy.int,
        exir_ops.edge.aten.permute_copy.default,
    }

    _VIEW_OPS = {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.view.default,
    }

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return list(self.transpose_or_permute_target)

    def maybe_remove_or_replace(self, node: Node) -> bool:
        parent_node = get_arg(node, "input", Node)

        # Case 1: Direct permute/transpose → permute/transpose
        if parent_node.target in self.transpose_or_permute_target:
            return self._fuse_direct(node, parent_node)

        # Case 2: permute → view_copy(squeeze/unsqueeze) → permute
        if parent_node.target in self._VIEW_OPS:
            return self._fuse_across_view(node, parent_node)

        return False

    def _fuse_direct(self, node: Node, parent_node: Node) -> bool:
        """Fuse two adjacent permute/transpose ops."""
        input_of_parent = get_arg(parent_node, "input", Node)
        dims = list(range(node.meta["val"].ndim))

        if parent_node.target == exir_ops.edge.aten.transpose_copy.int:
            dims = get_transposed_dims(parent_node, dims)
        else:
            dims = get_permuted_dims(parent_node, dims)

        if node.target == exir_ops.edge.aten.transpose_copy.int:
            dims = get_transposed_dims(node, dims)
        else:
            dims = get_permuted_dims(node, dims)

        if dims == sorted(dims):
            node.replace_all_uses_with(input_of_parent)
        else:
            with node.graph.inserting_before(node):
                new_permute = node.graph.call_function(
                    exir_ops.edge.aten.permute_copy.default,
                    args=(input_of_parent, dims),
                )
                new_permute.meta = node.meta
            node.replace_all_uses_with(new_permute)

        return True

    def _fuse_across_view(self, node: Node, view_node: Node) -> bool:
        """Fuse permute -> view(squeeze/unsqueeze) -> permute into a view_copy."""
        # view_node must have exactly one user (this permute node)
        if len(view_node.users) != 1:
            return False
        # view_node's parent must be a permute/transpose
        view_input = get_arg(view_node, "input", Node)
        if view_input.target not in self.transpose_or_permute_target:
            return False
        # The view must be a squeeze or unsqueeze (rank differs by 1)
        if view_node.meta.get("val") is None or view_input.meta.get("val") is None:
            return False
        view_in_shape = view_input.meta["val"].shape
        view_out_shape = view_node.meta["val"].shape
        if abs(len(view_in_shape) - len(view_out_shape)) != 1:
            return False

        # Get the input before the first permute
        input_of_first_permute = get_arg(view_input, "input", Node)
        if input_of_first_permute.meta.get("val") is None:
            return False

        # Compute the combined effect on the original input dimensions
        # Start with identity dims for the original input
        original_ndim = input_of_first_permute.meta["val"].ndim
        dims = list(range(original_ndim))

        # Apply first permute
        if view_input.target == exir_ops.edge.aten.transpose_copy.int:
            dims = get_transposed_dims(view_input, dims)
        else:
            dims = get_permuted_dims(view_input, dims)

        # Apply the view (squeeze/unsqueeze)
        if len(view_out_shape) == len(view_in_shape) + 1:
            # unsqueeze: insert a new dim
            index = self._find_extra_one(view_out_shape, view_in_shape)
            if index == -1:
                return False
            dims = [x + 1 if x >= index else x for x in dims]
            dims.insert(index, -1)  # -1 marks the inserted dim
        elif len(view_in_shape) == len(view_out_shape) + 1:
            # squeeze: remove a dim
            index = self._find_extra_one(view_in_shape, view_out_shape)
            if index == -1:
                return False
            if dims[index] != -1:
                # Safe: permutation preserves dimension sizes, so a size-1
                # intermediate dim necessarily originated from a size-1 input dim.
                pass
            del dims[index]

        # Apply second permute (node)
        if node.target == exir_ops.edge.aten.transpose_copy.int:
            node_dims = list(range(len(dims)))
            node_dims = get_transposed_dims(node, node_dims)
            dims = [dims[d] for d in node_dims]
        else:
            perm = get_arg(node, "dims")
            dims = [dims[d] for d in perm]

        # Check if the combined effect (ignoring -1 inserted dims) is identity
        real_dims = [d for d in dims if d != -1]

        if real_dims == sorted(real_dims):
            # Combined permutations are identity — replace with view_copy
            # (the only remaining effect is the squeeze/unsqueeze reshape)
            output_shape = node.meta["val"].shape
            if output_shape == input_of_first_permute.meta["val"].shape:
                # Total no-op: replace with input
                node.replace_all_uses_with(input_of_first_permute)
            else:
                with node.graph.inserting_before(node):
                    new_view = node.graph.call_function(
                        exir_ops.edge.aten.view_copy.default,
                        args=(input_of_first_permute, list(output_shape)),
                    )
                    new_view.meta = node.meta
                node.replace_all_uses_with(new_view)
            return True

        return False

    @staticmethod
    def _find_extra_one(longer, shorter):
        if len(longer) != len(shorter) + 1:
            return -1
        for i in range(len(shorter)):
            if longer[i] != shorter[i]:
                if longer[i] == 1 and shorter[i:] == longer[i + 1:]:
                    return i
                return -1
        return len(shorter) if longer[-1] == 1 else -1


from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class ValveView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "ValveView",
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        self._handles = RigidPrimView(prim_paths_expr="/World/envs/.*/valve/handle", name="handle_view", reset_xform_properties=False)
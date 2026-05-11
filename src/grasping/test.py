import mujoco
import numpy as np
import os

mesh_path = os.path.abspath("reconstructions/raspberry_mesh.obj")

xml = f"""
<mujoco>
    <asset>
        <mesh name="raspberry" file="{mesh_path}"/>
    </asset>
    <worldbody/>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

verts = model.mesh_vert.reshape(-1, 3)
print(f"num vertices: {len(verts)}")
print(f"x range: {verts[:,0].min():.4f} → {verts[:,0].max():.4f}")
print(f"y range: {verts[:,1].min():.4f} → {verts[:,1].max():.4f}")
print(f"z range: {verts[:,2].min():.4f} → {verts[:,2].max():.4f}")
print(f"centre:  {verts.mean(axis=0)}")
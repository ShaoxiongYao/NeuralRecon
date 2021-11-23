import open3d as o3d

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh("results/scene_demo_checkpoints_fusion_eval_47/HaomengHome2_raw.ply")

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print("Painting the mesh")
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])

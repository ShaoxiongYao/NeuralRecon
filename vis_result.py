import numpy as np
import open3d as o3d

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh("results/scene_demo_checkpoints_fusion_eval_47/RestAreawithPerson_mask_feature.ply")

    # RestAreawithPerson_mask_feature.ply
    # RestAreawithPerson_mask_image_mean.ply
    # RestAreawithPerson_mask_image_zero.ply
    # RestAreawithPerson_raw.ply

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
      
    # o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    theta = np.deg2rad(15)  
    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
#     mesh.paint_uniform_color([251/255.0,189/255.0,128/255.0])
    o3d.visualization.draw_geometries([mesh], 
                                          zoom=0.3,
                                          front=[0, -np.cos(theta), np.sin(theta)],
                                          lookat=[0.5, 0.5, 0.5],
                                          up=[0, 0, 1])

#     print("Painting the mesh")
#     mesh.paint_uniform_color([1, 0.706, 0])
#     o3d.visualization.draw_geometries([mesh])

import numpy as np
import open3d as o3d

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh("results/scene_demo_checkpoints_fusion_eval_47/slowPersonMove_mask_in_features.ply")

    # RestAreawithPerson_mask_feature.ply
    # RestAreawithPerson_mask_image_mean.ply
    # RestAreawithPerson_mask_image_zero.ply
    # RestAreawithPerson_raw.ply

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")      

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], 
                                      zoom=0.1,
                                      front=[ -0.99630886767291027, 0.046752485339995403, 0.071991980878825371 ],
                                      lookat=[ 0.50564805391588863, 0.72946279697650085, 1.1386682797845395 ],
                                      up=[ 0.071195739086218277, -0.018480705951851827, 0.99729114617722758 ])

# old visualization parameters
#     theta = np.deg2rad(15)  
#     o3d.visualization.draw_geometries([mesh], 
#                                       zoom=0.3,
#                                       front=[0, -np.cos(theta), np.sin(theta)],
#                                       lookat=[0.5, 0.5, 0.5],
#                                       up=[0, 0, 1])

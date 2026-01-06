#!/usr/bin/env python
"""
Hunyuan3D 2.1 Texture Generation Pipeline

Generates high-quality PBR textures for a mesh based on an input image and optional text prompt.
"""

import os
import sys
import time

# Apply torchvision compatibility fix before importing other modules
import torchvision_fix
torchvision_fix.apply_fix()

# Add module paths
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate textures for 3D mesh')
    parser.add_argument('--mesh', type=str, required=True, help='Path to input mesh (OBJ/GLB)')
    parser.add_argument('--image', type=str, required=True, help='Path to conditioning image')
    parser.add_argument('--prompt', type=str, default='high quality', help='Text prompt for texture style')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--max_views', type=int, default=6, help='Maximum number of views for multiview diffusion')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution for multiview generation')
    parser.add_argument('--no_glb', action='store_true', help='Skip GLB export (requires Blender)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Copy mesh to output directory for processing
    mesh_basename = os.path.basename(args.mesh)
    mesh_name = os.path.splitext(mesh_basename)[0]
    output_mesh_path = os.path.join(args.output, f"{mesh_name}_textured.obj")
    
    print(f"Loading texture generation pipeline...")
    print(f"  Mesh: {args.mesh}")
    print(f"  Image: {args.image}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Max views: {args.max_views}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Output: {output_mesh_path}")

    start_time = time.time()
    print(f"  Start time: {start_time}")
    
    # Import and configure the pipeline
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    
    # Create custom config
    config = Hunyuan3DPaintConfig(max_num_view=args.max_views, resolution=args.resolution)
    
    # Initialize pipeline
    print("\nInitializing Hunyuan3D Paint pipeline...")
    pipeline = Hunyuan3DPaintPipeline(config)
    
    # Override the hardcoded prompt in the pipeline
    # The pipeline uses image_caption = "high quality" internally
    # We'll need to modify the pipeline call or patch it
    
    # For now, let's use a patched version that accepts prompt
    from PIL import Image
    import trimesh
    import copy
    import numpy as np
    from utils.simplify_mesh_utils import remesh_mesh
    from utils.uvwrap_utils import mesh_uv_wrap
    
    # Load image
    image_prompt = Image.open(args.image)
    if not isinstance(image_prompt, list):
        image_prompt = [image_prompt]
    
    # Process mesh
    mesh_dir = args.output
    processed_mesh_path = os.path.join(mesh_dir, "white_mesh_remesh.obj")
    
    print(f"\nRemeshing input mesh...")
    remesh_mesh(args.mesh, processed_mesh_path)
    
    # Load mesh
    mesh = trimesh.load(processed_mesh_path)
    mesh = mesh_uv_wrap(mesh)
    pipeline.render.load_mesh(mesh=mesh)
    
    # View Selection
    print("Selecting views...")
    selected_camera_elevs, selected_camera_azims, selected_view_weights = pipeline.view_processor.bake_view_selection(
        config.candidate_camera_elevs,
        config.candidate_camera_azims,
        config.candidate_view_weights,
        config.max_selected_view_num,
    )
    
    # Render normal and position maps
    print("Rendering normal and position maps...")
    normal_maps = pipeline.view_processor.render_normal_multiview(
        selected_camera_elevs, selected_camera_azims, use_abs_coor=True
    )
    position_maps = pipeline.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
    
    # Style processing
    image_style = []
    for image in image_prompt:
        image = image.resize((512, 512))
        if image.mode == "RGBA":
            white_bg = Image.new("RGB", image.size, (255, 255, 255))
            white_bg.paste(image, mask=image.getchannel("A"))
            image = white_bg
        image_style.append(image)
    image_style = [image.convert("RGB") for image in image_style]
    
    # Multiview diffusion with custom prompt
    print(f"Running multiview diffusion with prompt: '{args.prompt}'...")
    multiviews_pbr = pipeline.models["multiview_model"](
        image_style,
        normal_maps + position_maps,
        prompt=args.prompt,  # Use custom prompt here!
        custom_view_size=config.resolution,
        resize_input=True,
    )
    
    # Enhance images
    print("Enhancing generated textures...")
    enhance_images = {}
    enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
    enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])
    
    for i in range(len(enhance_images["albedo"])):
        enhance_images["albedo"][i] = pipeline.models["super_model"](enhance_images["albedo"][i])
        enhance_images["mr"][i] = pipeline.models["super_model"](enhance_images["mr"][i])
    
    # Bake textures
    print("Baking textures...")
    for i in range(len(enhance_images["albedo"])):
        enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
            (config.render_size, config.render_size)
        )
        enhance_images["mr"][i] = enhance_images["mr"][i].resize((config.render_size, config.render_size))
    
    texture, mask = pipeline.view_processor.bake_from_multiview(
        enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
    )
    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    
    texture_mr, mask_mr = pipeline.view_processor.bake_from_multiview(
        enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
    )
    mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    
    # Inpaint
    print("Inpainting texture...")
    texture = pipeline.view_processor.texture_inpaint(texture, mask_np)
    pipeline.render.set_texture(texture, force_set=True)
    
    if "mr" in enhance_images:
        texture_mr = pipeline.view_processor.texture_inpaint(texture_mr, mask_mr_np)
        pipeline.render.set_texture_mr(texture_mr)
    
    # Save mesh
    print(f"Saving textured mesh to {output_mesh_path}...")
    pipeline.render.save_mesh(output_mesh_path, downsample=True)
    
    # Export GLB if not disabled
    if not args.no_glb:
        try:
            from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
            output_glb_path = output_mesh_path.replace(".obj", ".glb")
            if convert_obj_to_glb(output_mesh_path, output_glb_path):
                print(f"GLB exported to {output_glb_path}")
            else:
                print("GLB export failed (Blender may not be available)")
        except Exception as e:
            print(f"GLB export failed: {e}")
    
    print(f"\n✅ Texture generation complete!")
    print(f"   Output: {output_mesh_path}")
    time_taken = time.time() - start_time
    print(f"   Time taken: {time_taken:.2f} seconds")


if __name__ == "__main__":
    main()

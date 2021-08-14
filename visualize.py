import os
import cv2
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import convert_crop_cam_to_orig_img
from lib.core.config import DATA_DIR

#os.environ['PYOPENGL_PLATFORM'] = 'egl'

def main(args):
    data = joblib.load(args.input_file)
    data = {k: v[:16] for k, v in data.items()}
    num_images = len(data['paths'])

    renderer = Renderer(resolution = (args.width, args.height), orig_img=True,
        wireframe=args.wireframe)

    if args.upper_body:
        upper_body_indices = np.load(os.path.join(DATA_DIR, 'upper_body_indices.npy'))
        renderer.set_faces(upper_body_indices)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data['bboxes'][:, 2:] *= 1.3
    orig_cam = convert_crop_cam_to_orig_img(
        cam=data['pred_theta'][:, :3],
        bbox=data['bboxes'],
        img_width=args.width,
        img_height=args.height
    )
    for idx in tqdm(range(num_images)):
        output_path = os.path.join(args.output_dir, os.path.split(data['paths'][idx])[1])
        if os.path.exists(output_path):
            image = cv2.imread(output_path)
        else:
            image = cv2.imread(data['paths'][idx])
        render_image = renderer.render(image, data['pred_verts'][idx], orig_cam[idx])
        cx, cy, w, h = data['bboxes'][idx]
        left, right = int(cx - w / 2.0), int(cx + w / 2.0)
        top, bottom = int(cy - h / 2.0), int(cy + h / 2.0)
        # output_image = cv2.rectangle(render_image, (left, top), (right, bottom), (0, 0, 255), 3)

        # for kp in data['pred_j2d'][idx]:
        #     dx, dy = w * kp[0] / 2.0, h * kp[1] / 2.0
        #     cv2.circle(output_image, (int(cx+dx), int(cy+dy)), 3, (0, 0, 255), 4)
        cv2.imwrite(output_path, render_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input pickle file path', default='results/inference.pkl')
    parser.add_argument('--output_dir', type=str, help='output result directory', default='visual')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')
    parser.add_argument('--upper_body', action='store_true', help='only render upper body')
    args = parser.parse_args()

    main(args)

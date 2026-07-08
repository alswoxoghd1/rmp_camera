import argparse
import os

import cv2
import yaml

from rmp_camera.charuco_common import make_charuco_board


def main():
    parser = argparse.ArgumentParser(description="Generate a printable ChArUco board image.")
    parser.add_argument("--output", default="charuco_rb10_5x7.png")
    parser.add_argument("--config-output", default="charuco_rb10_5x7.yaml")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-length-m", type=float, default=0.04)
    parser.add_argument("--marker-length-m", type=float, default=0.03)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--pixels-per-meter", type=float, default=6000.0)
    parser.add_argument("--margin-px", type=int, default=120)
    parser.add_argument("--border-bits", type=int, default=1)
    args = parser.parse_args()

    board = make_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length_m,
        args.marker_length_m,
        args.dictionary,
    )

    board_width_px = int(round(args.squares_x * args.square_length_m * args.pixels_per_meter))
    board_height_px = int(round(args.squares_y * args.square_length_m * args.pixels_per_meter))
    image_size = (
        board_width_px + 2 * args.margin_px,
        board_height_px + 2 * args.margin_px,
    )
    image = board.draw(image_size, marginSize=args.margin_px, borderBits=args.border_bits)

    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if not cv2.imwrite(output, image):
        raise RuntimeError(f"Failed to write board image: {output}")

    config = {
        "dictionary": args.dictionary,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_m": args.square_length_m,
        "marker_length_m": args.marker_length_m,
        "printed_width_m": args.squares_x * args.square_length_m,
        "printed_height_m": args.squares_y * args.square_length_m,
        "notes": "Print at 100% scale. Do not fit-to-page. Verify printed square size with a ruler.",
    }
    config_output = os.path.abspath(os.path.expanduser(args.config_output))
    os.makedirs(os.path.dirname(config_output) or ".", exist_ok=True)
    with open(config_output, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(f"Wrote ChArUco board: {output}")
    print(f"Wrote board config: {config_output}")
    print(
        "Printed board size: "
        f"{config['printed_width_m'] * 1000.0:.1f} mm x "
        f"{config['printed_height_m'] * 1000.0:.1f} mm"
    )
    print("Print at 100% scale and measure one square. It must be exactly "
          f"{args.square_length_m * 1000.0:.1f} mm.")


if __name__ == "__main__":
    main()

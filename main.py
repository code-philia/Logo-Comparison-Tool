from matcher import LogoMatcher, display_matches
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_img", required=True, type=str)
    parser.add_argument("--sim_thresh", default=0.87, type=float)
    parser.add_argument("--visualize_path", default="./test_result.png", type=str)
    args = parser.parse_args()

    matcher = LogoMatcher(ref_dir = './models/expand_targetlist',
                          model_weights_path = './models/resnetv2_rgb_new.pth.tar',
                          sim_thresh = args.sim_thresh)

    matches = matcher(query_logo=args.query_img)
    if len(matches):
        display_matches(args.query_img, matches)
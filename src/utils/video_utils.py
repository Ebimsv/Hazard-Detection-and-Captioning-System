import os
import pickle


class VideoProcessor:
    def __init__(self, annotations_path, video_root):
        self.annotations_path = annotations_path
        self.video_root = video_root
        self.annotations = self.load_annotations()

    def load_annotations(self):
        with open(self.annotations_path, "rb") as f:
            annotations = pickle.load(f)
        return annotations

    def check_videos_exist(self):
        for video in self.annotations.keys():
            assert os.path.exists(os.path.join(self.video_root, video + ".mp4"))

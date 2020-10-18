# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Welcome to torchvision's new video API
#
# Here, we're going to examine the capabilities of the new video API, together with the examples on how to build datasets and more.
#
# ### Table of contents
# 1. Introduction: building a new video object and examining the properties
# 2. Building a sample `read_video` function
# 3. Building an example dataset (can be applied to e.g. kinetics400)

import torch
import torchvision
torch.__version__, torchvision.__version__

# download the sample video
from torchvision.datasets.utils import download_url
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/WUzgd7C1pWA.mp4?raw=true", ".", "WUzgd7C1pWA.mp4")

# ## 1. Introduction: building a new video object and examining the properties
#
# First we select a video to test the object out. For the sake of argument we're using one from Kinetics400 dataset. To create it, we need to define the path and the stream we want to use. See inline comments for description.

# +
import torch
import torchvision
"""
chosen video statistics:
WUzgd7C1pWA.mp4
  - source: kinetics-400
  - video: H-264 - MPEG-4 AVC (part 10) (avc1)
    - fps: 29.97
  - audio: MPEG AAC audio (mp4a)
    - sample rate: 48K Hz
"""
video_path = "./WUzgd7C1pWA.mp4"

"""
streams are defined in a similar fashion as torch devices. We encode them as strings in a form
of `stream_type:stream_id` where stream_type is a string and stream_id a long int. 

The constructor accepts passing a stream_type only, in which case the stream is auto-discovered.
"""
stream = "video"


video = torchvision.io.VideoReader(video_path, stream)
# -

# First, let's get the metadata for our particular video:

video.get_metadata()

# Here we can see that video has two streams - a video and an audio stream.
#
# Let's read all the frames from the video stream.

# +
# first we select the video stream
metadata = video.get_metadata()
video.set_current_stream("video:0")

frames = []  # we are going to save the frames here.
for frame, pts in video:
    frames.append(frame)

print("Total number of frames: ", len(frames))
approx_nf = metadata['video']['duration'][0] * metadata['video']['fps'][0]
print("We can expect approx: ", approx_nf)
print("Tensor size: ", frames[0].size())
# -

# Note that selecting zero video stream is equivalent to selecting video stream automatically. I.e. `video:0` and `video` will end up with same results in this case.
#
# Let's try this for audio

# +
metadata = video.get_metadata()
video.set_current_stream("audio")

frames = []  # we are going to save the frames here.
for frame, pts in video:
    frames.append(frame)

print("Total number of frames: ", len(frames))
approx_nf = metadata['audio']['duration'][0] * metadata['audio']['framerate'][0]
print("Approx total number of datapoints we can expect: ", approx_nf)
print("Read data size: ", frames[0].size(0) * len(frames))
# -

# But what if we only want to read certain time segment of the video?
#
# That can be done easily using the combination of our seek function, and the fact that each call to next returns the presentation timestamp of the returned frame in seconds. Given that our implementation relies on python iterators, we can leverage `itertools` to simplify the process and make it more pythonic.
#
# For example, if we wanted to read ten frames from second second:

# +
import itertools
video.set_current_stream("video")

frames = []  # we are going to save the frames here.

# we seek into a second second of the video
# and use islice to get 10 frames since
for frame, pts in itertools.islice(video.seek(2), 10):
    frames.append(frame)

print("Total number of frames: ", len(frames))
# -

# Or if we wanted to read from 2nd to 5th second:

# +
video.set_current_stream("video")

frames = []  # we are going to save the frames here.

# we seek into a second second of the video
video = video.seek(2)
# then we utilize the itertools takewhile to get the
# correct number of frames
for frame, pts in itertools.takewhile(lambda x: x[1] <= 5, video):
    frames.append(frame)

print("Total number of frames: ", len(frames))
approx_nf = (5 - 2) * video.get_metadata()['video']['fps'][0]
print("We can expect approx: ", approx_nf)
print("Tensor size: ", frames[0].size())


# -

# ## 2. Building a sample `read_video` function
#
# We can utilize the methods above to build the read video function that follows the same API to the existing `read_video` function

def example_read_video(video_object, start=0, end=None, read_video=True, read_audio=True):

    if end is None:
        end = float("inf")
    if end < start:
        raise ValueError(
            "end time should be larger than start time, got "
            "start time={} and end time={}".format(s, e)
        )

    video_frames = torch.empty(0)
    video_pts = []
    if read_video:
        video_object.set_current_stream("video")
        frames = []
        for t, pts in itertools.takewhile(lambda x: x[1] <= end, video_object.seek(start)):
            frames.append(t)
            video_pts.append(pts)
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)

    audio_frames = torch.empty(0)
    audio_pts = []
    if read_audio:
        video_object.set_current_stream("audio")
        frames = []
        for t, pts in itertools.takewhile(lambda x: x[1] <= end, video_object.seek(start)):
            frames.append(t)
            video_pts.append(pts)
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)

    return video_frames, audio_frames, (video_pts, audio_pts), video_object.get_metadata()


vf, af, info, meta = example_read_video(video)
# total number of frames should be 327 for video and 523264 datapoints for audio
print(vf.size(), af.size())

# you can also get the sequence of audio frames as well
af.size()

# ## 3. Building an example randomly sampled dataset (can be applied to training dataest of kinetics400)
#
# Cool, so now we can use the same principle to make the sample dataset. We suggest trying out iterable dataset for this purpose.
#
# Here, we are going to build
#
# a. an example dataset that reads randomly selected 10 frames of video

# make sample dataest
import os
os.makedirs("./dataset", exist_ok=True)
os.makedirs("./dataset/1", exist_ok=True)
os.makedirs("./dataset/2", exist_ok=True)

# download the videos
from torchvision.datasets.utils import download_url
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/WUzgd7C1pWA.mp4?raw=true",
             "./dataset/1", "WUzgd7C1pWA.mp4")
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi?raw=true",
             "./dataset/1", "RATRACE_wave_f_nm_np1_fr_goo_37.avi")
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/SOX5yA1l24A.mp4?raw=true",
             "./dataset/2", "SOX5yA1l24A.mp4")
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/v_SoccerJuggling_g23_c01.avi?raw=true",
             "./dataset/2", "v_SoccerJuggling_g23_c01.avi")
download_url("https://github.com/pytorch/vision/blob/master/test/assets/videos/v_SoccerJuggling_g24_c01.avi?raw=true",
             "./dataset/2", "v_SoccerJuggling_g24_c01.avi")

# +
# housekeeping and utilities
import os
import random

import torch
from torchvision.datasets.folder import make_dataset
from torchvision import transforms as t


def _find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_samples(root, extensions=(".mp4", ".avi")):
    _, class_to_idx = _find_classes(root)
    return make_dataset(root, class_to_idx, extensions=extensions)


# -

# We are going to define the dataset and some basic arguments. We asume the structure of the FolderDataset, and add the following parameters:
#
# 1. frame transform: with this API, we can chose to apply transforms on every frame of the video
# 2. videotransform: equally, we can also apply transform to a 4D tensor
# 3. length of the clip: do we want a single or multiple frames?
#
# Note that we actually add `epoch size` as using `IterableDataset` class allows us to naturally oversample clips or images from each video if needed.

class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self, root, epoch_size=None, frame_transform=None, video_transform=None, clip_len=16):
        super(RandomDataset).__init__()

        self.samples = get_samples(root)

        # allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.clip_len = clip_len  # length of a clip in frames
        self.frame_transform = frame_transform  # transform for every frame individually
        self.video_transform = video_transform  # transform on a video sequence

    def __iter__(self):
        for i in range(self.epoch_size):
            # get random sample
            path, target = random.choice(self.samples)
            # get video object
            vid = torchvision.io.VideoReader(path, "video")
            metadata = vid.get_metadata()
            video_frames = []  # video frame buffer
            # seek and return frames

            max_seek = metadata["video"]['duration'][0] - (self.clip_len / metadata["video"]['fps'][0])
            start = random.uniform(0., max_seek)
            for frame, current_pts in itertools.islice(vid.seek(start), self.clip_len):
                video_frames.append(self.frame_transform(frame))
            # stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)
            output = {
                'path': path,
                'video': video,
                'target': target,
                'start': start,
                'end': current_pts}
            yield output


# Given a path of videos in a folder structure, i.e:
# ```
# dataset:
#     -class 1:
#         file 0
#         file 1
#         ...
#     - class 2:
#         file 0
#         file 1
#         ...
#     - ...
# ```
# We can generate a dataloader and test the dataset.
#

# +
from torchvision import transforms as t
transforms = [t.Resize((112, 112))]
frame_transform = t.Compose(transforms)

ds = RandomDataset("./dataset", epoch_size=None, frame_transform=frame_transform)
# -

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=12)
d = {"video": [], 'start': [], 'end': [], 'tensorsize': []}
for b in loader:
    for i in range(len(b['path'])):
        d['video'].append(b['path'][i])
        d['start'].append(b['start'][i].item())
        d['end'].append(b['end'][i].item())
        d['tensorsize'].append(b['video'][i].size())

d

# Cleanup
import os
import shutil
os.remove("./WUzgd7C1pWA.mp4")
shutil.rmtree("./dataset")

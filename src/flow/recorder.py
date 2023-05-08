"""User defined logging and analytics utility for training runs"""
import logging
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from shs.recorder_base import RecorderBase
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from astropy.visualization import ZScaleInterval


def zscale(data, contrast=0.2):
    norm = ZScaleInterval(contrast=contrast)
    return norm(data)


class Recorder(RecorderBase):
    """Artifact, metric, parameter, and image logger. Define custom analytic logic here."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.epoch = 0

    def custom_figure_creation(self):
        raise NotImplementedError

    def custom_metric_computations(self):
        raise NotImplementedError

    def log_dataframe(self, dataframe, artifact_path=None):
        self.client.log_text(self.run_id, dataframe.to_csv(index=False), artifact_path)

    def log_json(self, json_dict, artifact_path):
        self.client.log_text(
            self.run_id, json.dumps(json_dict, indent=4), artifact_path
        )

    def log_deep_image_grid(
        self,
        x: torch.Tensor,
        name: str = "x",
        NCHW: bool = True,
        normalize: bool = True,
        jpg: bool = True,
        padding: bool = 1,
    ):
        """log batch of images

        Args:
            x: torch.float32 of shape (N, C, H, W) or (N, H, W, C)
            name: filename of rendered image, not including file extension
            NCHW: if false, will convert from NHWC to NCHW
            normalize: apply per-instance normalization
            jpg: if false, convert to png
            padding: pixel padding between images in grid
        """
        # convert NCHW to N 1 H*sqrt(C) W*sqrt(C)
        #
        pass

    def log_star_fits(self, samples, targets, outputs):
        print(samples)
        print(targets)
        print(outputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        out_logits, out_line = outputs["pred_logits"], outputs["pred_lines"]
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        orig_size = torch.as_tensor([512, 512])
        img_h, img_w = orig_size.unbind(0)
        scale_fct = torch.unsqueeze(
            torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0
        )
        lines = out_line * scale_fct[:, None, :]
        lines = lines.view(1000, 2, 2)
        lines = lines.flip([-1])  # this is yxyx format
        scores = scores.detach().numpy()
        keep = scores >= 0.6
        keep = keep.squeeze()
        lines = lines[keep]
        lines = lines.reshape(lines.shape[0], -1)

        fig = plt.figure()
        plt.imshow(zscale(samples[-1]))

        for tp_id, line in enumerate(lines):
            y1, x1, y2, x2 = line.detach().numpy()  # this is yxyx

            # x1, y1, x2, y2  = line.detach().numpy()
            p1 = (x1, y1)
            p2 = (x2, y2)
            plt.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.25, color="red", zorder=1
            )

        plt.axis("off")

        figure = self.figure_to_array(fig)
        self.client.log_image(self.run_id, figure, "%s_conf_matrix.png" % self.epoch)
        plt.close(fig)

    def log_confusion_matrix(
        self, y_full, y_pred_full, num_classes, slice="val", classnames=None
    ):
        conf_mat = np.zeros((num_classes, num_classes))
        for idx in range(len(y_full)):
            conf_mat[y_full[idx], np.argmax(y_pred_full[idx])] += 1
        conf_mat /= conf_mat.sum(axis=1, keepdims=True)

        extra_args = {}
        if classnames is not None:
            extra_args = {
                "xticklabels": classnames,
                "yticklabels": classnames,
            }
        fig, ax = plt.subplots(figsize=(num_classes * 2 / 3, num_classes * 2 / 3))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            linecolor="black",
            cmap="gray",
            **extra_args
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        # plt.show(block=False)

        figure = self.figure_to_array(fig)
        self.client.log_image(self.run_id, figure, "%s_conf_matrix.png" % slice)
        plt.close(fig)

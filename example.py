from src.models.caffe.Caffe2Model import Caffe2Model
import src.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def main():
    image_file_name = "dataset/test/exp9_zebra_fish0004.jpg"
    weight_folder_path = "weights/caffe2/unlabeled_fish_cpu"

    model = Caffe2Model.load_protobuf(weight_folder_path)
    image_bgr = utils.read_image(image_file_name, format="BGR")


    image = np.array(np.einsum("hwc->chw", image_bgr))


    prediction = model(image)
    print(prediction.pred_masks.shape)
    print(prediction)

    plot_item(image_bgr, prediction)
    print(prediction.pred_masks.sum(axis=0).shape)



def plot_item(image, predicion):
    image = np.array(image)

    ax = plt.axes()
    ax.imshow(image)

    for xmin, ymin, xmax, ymax in predicion.pred_boxes:
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.imshow(predicion.pred_masks.sum(axis=0) > 0, 'gray', interpolation='none', alpha=.5)
    plt.show()

if __name__ == '__main__':
    main()
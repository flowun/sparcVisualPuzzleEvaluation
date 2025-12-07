
# this script uses multiprocessing to turn the full dataset into images and saves them
import os
from multiprocessing import Pool
from datasets import load_from_disk, load_dataset
from objects import board
from plots.plot import get_plot_class


def create_board_image(data, split_savename="test", subset_savename="all", plot_type="original", kwargs=None):
    if kwargs is None:
        kwargs = {}
    b = board.get_board_from_data(data)
    plot = get_plot_class(plot_type, b, size=(b.width, b.height), **kwargs)
    plot.render()
    plot.save(dir=f"data/boards/{plot_type}/{split_savename}/{subset_savename}", filename=data['id'] + ".png")

def create_board_images_in_parallel(dataset, split_savename="test", subset_savename="all", plot_type="original", processes=os.cpu_count() // 2, **kwargs):
    with Pool(processes=max(1, processes or 1)) as pool:
        pool.starmap(create_board_image, [(data,
                                           split_savename,
                                           subset_savename,
                                           plot_type,
                                           kwargs) for data in dataset], chunksize=1)

if __name__ == "__main__":
    # test_dataset = load_from_disk("data/sparc_text_dataset")["test"]

    split = "test"
    subset = "all"
    plot_type = "black_frame_and_path_cell_annotated"

    dataset = load_dataset("lkaesberg/SPaRC", subset, split=split)
    create_board_images_in_parallel(dataset, split_savename=split, subset_savename=subset, plot_type=plot_type)
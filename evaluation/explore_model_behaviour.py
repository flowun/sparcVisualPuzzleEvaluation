import os
import json

from datasets import load_dataset

from objects.board import get_board_from_data
from plots.plot import get_plot_class

model = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
board_type = "original"
prompt_type = "default_tr"
subset = "all"
split = "test"
evaluation_file = "original-B_default_tr-P_20251127_0427_stats_individual"
evaluation_file_path = f"results/{split}/{subset}/{model.split('/')[-1]}/{evaluation_file}"
dataset_revision = "195579019ab44fce4f394bb03af04bf598956e4b"
tmp_folder = "../data/.tmp"
"""
GUI that visualized model outputs:
- left panel (mid and top): show puzzle image with overlayed path from model
- left panel (bottom): buttons to navigate: previous/next puzzle, thoughts or answer button and whether the solution is correct
- right panel: model output (can be switched between everything before </think> and everything after)
"""
# Use Pillow for PNG loading in Tkinter
from PIL import Image, ImageTk

def load_evaluation_results(evaluation_file_path):
    with open(f"{evaluation_file_path}.json", "r", encoding="utf-8") as f:
        eval_results = json.load(f)
    return eval_results

def get_visualization_data(dataset, eval_results, idx):
    id = dataset[idx]['id']
    eval_result = eval_results.get(str(id), None)
    if eval_result is None:
        raise ValueError(f"No evaluation data found for id: {id}")
    # Robust correctness and output parsing
    correct_solution = bool(eval_result.get('is_valid', False))
    model_output = eval_result.get('response', '') or ''
    print(f"Model output for id {id}")
    # Split around </think>: left = thoughts (including any <think>), right = after-think content
    if '</think>' in model_output:
        before, after = model_output.split('</think>', 1)
        thinking_output = before
        answer_output = after
    else:
        # Fallback: if no closing tag, treat everything as "answer"
        thinking_output = ""
        answer_output = model_output # "Did not finish thinking (missing </think>)."

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    img_path = f"{tmp_folder}/{id}.png"
    if not os.path.exists(img_path):
        b = get_board_from_data(dataset[idx])
        plot = get_plot_class(plot_type=board_type, board=b, size=(b.width, b.height))
        plot.render()
        try:
            plot.overlay_path(eval_result['extracted_path'])
        except Exception as e:
            pass
        plot.save(dir=tmp_folder, filename=f"{id}.png")
    return img_path, thinking_output, answer_output, correct_solution

def cleanup_tmp_folder(tmp_folder):
    if os.path.exists(tmp_folder):
        for file in os.listdir(tmp_folder):
            file_path = os.path.join(tmp_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

class ModelBehaviourExplorerGUI:
    def __init__(self, dataset, eval_results):
        self.dataset = dataset
        self.eval_results = eval_results
        self.current_index = 0
        # store current PIL image to prevent GC of PhotoImage
        self._current_tk_img = None
        self._current_img_path = None  # track current image path for resize re-render

    def run(self):
        # Build split-pane GUI with image on left and text on right
        from tkinter import Tk, Label, Button, Text, Scrollbar, RIGHT, LEFT, Y, BOTH, END, TOP, BOTTOM, Frame, PanedWindow, Radiobutton, StringVar, X
        from tkinter import font as tkfont

        def load_and_set_image(img_path):
            # Scale image to fit available left pane space (above controls)
            try:
                img = Image.open(img_path)
            except Exception:
                # If we cannot open the image, show a placeholder and clear old image
                show_placeholder(mode='missing')
                return
            # Compute max size for the image area
            left_frame.update_idletasks()
            # Available width = left_frame width
            max_w = max(50, left_frame.winfo_width())
            # Available height = left_frame height - controls min height
            controls_h = controls_frame.winfo_height() if controls_frame.winfo_height() > 0 else controls_min_height
            max_h = max(50, left_frame.winfo_height() - controls_h)
            # Resize with aspect ratio
            img_copy = img.copy()
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            img_copy.thumbnail((max_w, max_h), resample)
            tk_img = ImageTk.PhotoImage(img_copy)
            image_label.config(image=tk_img, text="")  # clear any placeholder text
            image_label.image = tk_img
            self._current_tk_img = tk_img  # keep reference

        def update_visualization():
            # Immediately clear old image and show a loading placeholder
            show_placeholder(mode='loading')
            try:
                img_path, thinking_output, answer_output, correct_solution = get_visualization_data(
                    self.dataset, self.eval_results, self.current_index)
            except Exception as e:
                # In case of missing data, show error and placeholder
                thinking_output = ""
                answer_output = f"Error: {e}"
                correct_solution = False
                img_path = None

            if img_path and os.path.exists(img_path):
                self._current_img_path = img_path
                load_and_set_image(img_path)
            else:
                # No image available: ensure old image is removed
                show_placeholder(mode='missing')

            text_box.config(state='normal')
            text_box.delete(1.0, END)
            if output_mode.get() == 'thoughts':
                text_box.insert(END, thinking_output.strip())
            else:
                text_box.insert(END, answer_output.strip())
            text_box.config(state='normal')

            index_label.config(text=f"Puzzle {self.current_index + 1} / {len(self.dataset)}")
            if correct_solution:
                correctness_label.config(text="Correct ✓", fg="green")
            else:
                correctness_label.config(text="Incorrect ✗", fg="red")

        def next_puzzle():
            if self.current_index < len(self.dataset) - 1:
                self.current_index += 1
                update_visualization()

        def previous_puzzle():
            if self.current_index > 0:
                self.current_index -= 1
                update_visualization()

        def toggle_output():
            update_visualization()

        root = Tk()
        root.title("Model Behaviour Explorer")

        # Start fullscreen/maximized
        try:
            root.state('zoomed')  # Windows
        except Exception:
            try:
                root.attributes('-zoomed', True)  # X11/Linux
            except Exception:
                root.attributes('-fullscreen', True)  # Fallback

        # Define larger fonts for controls
        btn_font = tkfont.Font(size=14, weight="bold")
        radio_font = tkfont.Font(size=13)
        label_font = tkfont.Font(size=12)
        placeholder_font = tkfont.Font(size=48, weight="bold")  # for '?' or '…'

        # Split into left and right panes
        paned = PanedWindow(root, orient='horizontal')
        paned.pack(fill=BOTH, expand=True)

        # Left pane: image (top/mid) and controls (bottom)
        left_frame = Frame(paned)

        paned.add(left_frame, minsize=650)

        # Use grid to reserve space for controls
        controls_min_height = 110  # px; guarantees large buttons area
        left_frame.grid_rowconfigure(0, weight=1)  # image area grows
        left_frame.grid_rowconfigure(1, weight=0, minsize=controls_min_height)  # controls fixed min height
        left_frame.grid_columnconfigure(0, weight=1)

        image_label = Label(left_frame)
        image_label.grid(row=0, column=0, sticky="nsew")

        # Helper to clear any previous image and show a placeholder
        def show_placeholder(mode='missing'):
            # Reset current image path so resize does not reload a stale image
            self._current_img_path = None
            self._current_tk_img = None
            image_label.config(image="", text=("…" if mode == 'loading' else "?"), font=placeholder_font)
            image_label.image = None

        controls_frame = Frame(left_frame, height=controls_min_height)
        controls_frame.grid(row=1, column=0, sticky="ew")
        controls_frame.grid_propagate(False)  # keep reserved height even when image is tall

        prev_button = Button(controls_frame, text="Previous", command=previous_puzzle, font=btn_font)
        prev_button.pack(side=LEFT)
        prev_button.config(padx=14, pady=8)

        next_button = Button(controls_frame, text="Next", command=next_puzzle, font=btn_font)
        next_button.pack(side=LEFT)
        next_button.config(padx=14, pady=8)

        # Thoughts/Answer toggle
        output_mode = StringVar(value='thoughts')
        thoughts_radio = Radiobutton(controls_frame, text="Thoughts", variable=output_mode, value='thoughts', command=toggle_output, font=radio_font)
        thoughts_radio.pack(side=LEFT)
        answer_radio = Radiobutton(controls_frame, text="Answer", variable=output_mode, value='answer', command=toggle_output, font=radio_font)
        answer_radio.pack(side=LEFT)

        # Correctness indicator and index
        correctness_label = Label(controls_frame, text="", fg="green", font=label_font)
        correctness_label.pack(side=LEFT, padx=10)
        index_label = Label(controls_frame, text="", font=label_font)
        index_label.pack(side=LEFT, padx=10)

        # Re-render image when left pane resizes (keeps it contained, buttons stay large)
        def on_left_resize(event):
            if self._current_img_path and os.path.exists(self._current_img_path):
                load_and_set_image(self._current_img_path)
        left_frame.bind("<Configure>", on_left_resize)

        # Right pane: model output with scrollbar
        right_frame = Frame(paned)
        paned.add(right_frame)

        text_container = Frame(right_frame)
        text_container.pack(side=TOP, fill=BOTH, expand=True)

        scrollbar = Scrollbar(text_container)
        scrollbar.pack(side=RIGHT, fill=Y)

        text_box = Text(text_container, wrap='word', yscrollcommand=scrollbar.set)
        text_box.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=text_box.yview)

        # Ensure initial sash placement gives the left pane enough width
        def set_initial_layout():
            try:
                paned.sash_place(0, root.winfo_screenwidth() * 0.45, 1)
            except Exception:
                pass
        root.after(100, set_initial_layout)

        # Initial render
        update_visualization()
        root.mainloop()

if __name__ == "__main__":
    dataset = load_dataset("lkaesberg/SPaRC", subset, split=split, revision=dataset_revision)
    eval_results = load_evaluation_results(evaluation_file_path)
    gui = ModelBehaviourExplorerGUI(dataset, eval_results)
    gui.run()
    cleanup_tmp_folder(tmp_folder)

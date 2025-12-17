import matplotlib.pyplot as plt
import os

def create_bar_chart(
    data,
    title="Bar Chart",
    x_label="Categories",
    y_label="Accuracy (%)",
    output_path='',
    bar_color='blue',
    highlighted_bars=None,
    highlighted_bar_color='orange',
    figsize=(10, 6),
    save_format='pdf',
    spines=False,
    x_grid=False,
    y_grid=True,
    transparent=False,
    text_on_bars=True,
    y_limit=None,
    title_fontsize=14,
):
    categories = [category for category in data.keys()]
    values = [value * 100 for value in data.values()]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

    bars = ax.bar(categories, values, color=bar_color, width=0.6)

    if highlighted_bars:
        for i, category in enumerate(categories):
            print("Checking category for highlighting:", category, i)
            print(highlighted_bars)
            if category in highlighted_bars:
                print("Highlighting bar:", category, i)
                bars[i].set_color(highlighted_bar_color)

    ax.set_ylim(0, max(max(values) * 1.1, y_limit if y_limit else 0))

    if not spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_title(title, fontsize=title_fontsize)

    if x_grid:
        ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='0.75', alpha=0.6)
    if y_grid:
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='0.75', alpha=0.6)

    if text_on_bars:
        for i, height in enumerate(values):
            ax.text(i, height + max(values) * 0.02, f'{round(height, 1)}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, transparent=transparent, bbox_inches='tight', pad_inches=0, format=save_format)

    plt.subplots_adjust(right=0.9)
    plt.show()
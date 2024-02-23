import matplotlib.pyplot as plt
import pandas as pd
import os

def visclassdist(dif_cl, occs, title, plot_path=None, filename=None):
    fig, ax = plt.subplots()
    ax.bar(dif_cl, occs, color="deepskyblue")
    ax.set_ylim(0, max(occs) + 1000)
    ax.set_xlabel('Classes')
    ax.set_ylabel('Occurrences')
    ax.set_title(title)
    for i, value in enumerate(occs):
        ax.text(i, value + 500, str(value), ha='center', va='top')
    
    plt.tight_layout()

    if plot_path:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        filename = filename if filename else title.replace(' ', '_').replace('.', '').replace('\n', '_') + '.png'
        full_path = os.path.join(plot_path, filename)
        fig.savefig(full_path)
        print(f"Plot saved to: {full_path}")
    else:
        plt.show()

    # Print a table of class distribution
    df_dict = {'class_names': dif_cl, 'occurrences': occs}
    print(pd.DataFrame(data=df_dict))

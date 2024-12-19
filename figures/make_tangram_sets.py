import matplotlib.pyplot as plt
import matplotlib.image as mpimg

tangram_sets = {
    0: ["A", "B", "C", "D", "H", "L"],
    1: ["E", "F", "G", "I", "J", "K"],
    2: ["A", "C", "E", "G", "I", "K"]
}

def create_tangram_grid(set_id, tangrams):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.suptitle(f"Tangram set {set_id}", fontsize=16)
    
    for i, tangram in enumerate(tangrams):
        row, col = divmod(i, 3)
        img_path = f"tangrams/tangram_{tangram}.png"
        
        # Load the image
        img = mpimg.imread(img_path)
        
        # Display the image in the grid
        axes[row, col].imshow(img)
        axes[row, col].set_title(tangram, fontsize=16)
        axes[row, col].axis("off")
    
    # Hide any unused subplots
    for i in range(len(tangrams), 6):
        row, col = divmod(i, 3)
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust space for the title
    plt.savefig(f"tangram_set_{set_id}.png")

for set_id, tangrams in tangram_sets.items():
    create_tangram_grid(set_id, tangrams)
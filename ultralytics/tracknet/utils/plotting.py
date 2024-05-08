from matplotlib import patches, patheffects, pyplot as plt
import numpy as np

# check_training_img_path = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img\img_'
# check_training_img_path = r'/usr/src/datasets/tracknet/visualize_train_img/img_'
check_training_img_path = "./test_data/"

def display_predict_in_checkerboard(target, pred, fileName, input_number=None):
    x, y, dx, dy, hit = target[0]

    # Calculate the range to display based on the current position
    x_min = max(x // 32 * 32 - 32, 0)
    x_max = x // 32 * 32 + 32 * 2
    y_min = max(y // 32 * 32 - 32, 0)
    y_max = y // 32 * 32 + 32 * 2
    
    # Create a plot
    plt.figure(figsize=(6, 6))

    # Determine grid lines within the specified range
    grid_lines_x = np.arange(x_min, x_max+1, 1)
    grid_lines_y = np.arange(y_min, y_max+1, 1)
    line_widths_x = [2 if line % 32 == 0 else 0.5 for line in grid_lines_x]
    line_widths_y = [2 if line % 32 == 0 else 0.5 for line in grid_lines_y]

    if hit == 1:
        hit_xy = (x, y)
        plt.scatter(*hit_xy, color='red', s=6)
    plot_x(x, y, 1.3, 'red', 'gc')
    plot_x(x+dx, y+dy, 0.8, 'pink', 'gn')

    # Plotting the predictions
    i = 0
    for (x_coordinates, y_coordinates, x, y, dx, dy, conf, hit) in pred:
        x_coordinates *= 32
        y_coordinates *= 32
        current_x = x_coordinates + x * 32
        current_y = y_coordinates + y * 32
        next_x = current_x + dx * 640
        next_y = current_y + dy * 640

        if float(hit) >= 0.5:
            hit_xy = (current_x, current_y)
            plt.scatter(*hit_xy, color='blue', s=6)
            plot_x(current_x, current_y, 1, 'blue', f'pc{i}: {conf}, h: {hit}')
        else:
            plot_x(current_x, current_y, 1, 'blue', f'pc{i}: {conf}')
        plot_x(next_x, next_y, 0.5, (0.34, 0.425, 0.95), f'pn{i}')
        i+=1

    # Adding grid lines with custom widths
    for i, line in enumerate(grid_lines_x):
        plt.axvline(x=line, color='gray', linewidth=line_widths_x[i], zorder=0)
    for i, line in enumerate(grid_lines_y):
        plt.axhline(y=line, color='gray', linewidth=line_widths_y[i], zorder=0)

    plt.gca().invert_yaxis()
    # Adding labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Prediction Visualization in Checkerboard')

    # Adding a legend
    #plt.legend(loc='upper right')

    # Set the limits for x and y to only show the relevant area
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Save the plot to a file
    plt.savefig(os.path.join(check_training_img_path, fileName), dpi=200)
    plt.close()

def plot_x(x, y, linewidth, color, label):
    size = 0.5
    x_values1 = [x - size, x + size]
    y_values1 = [y - size, y + size]
    x_values2 = [x - size, x + size]
    y_values2 = [y + size, y - size]
    plt.plot(x_values1, y_values1, c=color, linewidth=linewidth)
    plt.plot(x_values2, y_values2, c=color, linewidth=linewidth)
    plt.text(x+1, y+1, label, fontsize=5)

def display_image_with_coordinates(img_tensor, target, pred, fileName, input_number = None, img_filepath=''):
    
    # Convert the image tensor to numpy array
    img_array = img_tensor.cpu().numpy()

    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img_array, cmap='gray')

    img_height, img_width = img_array.shape[:2]

    # Plot each coordinate

    # 紅色跟粉紅色 -> Ground Truth
    for (x, y, dx, dy) in target:
        cell_x = x//32*32
        cell_y = (y//32)*32
        rect = patches.Rectangle(xy=(cell_x, cell_y), height=32, width=32, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(x, y, s=1.8, c='red', marker='o')
        #ax.scatter(x+dx, y+dy, s=1.6, c='#FFC0CB', marker='o')

    for (x_coordinates, y_coordinates, x, y, dx, dy, conf, hit) in pred:
        x_coordinates *= 32
        y_coordinates *= 32
        current_x = x_coordinates+x*32
        current_y = y_coordinates+y*32
        next_x = current_x+dx*640
        next_y = current_y+dy*640
        rect = patches.Rectangle(xy=(x_coordinates, y_coordinates), height=32, width=32, edgecolor='blue', facecolor='none', linewidth=0.5)
        ax.add_patch(rect)
        text = ax.text(x_coordinates+32+1, y_coordinates+32, str(conf), verticalalignment='bottom', horizontalalignment='left', fontsize=5)
        text.set_path_effects([patheffects.Stroke(linewidth=2, foreground=(1, 1, 1, 0.3)),
                       patheffects.Normal()])
        ax.scatter(current_x, current_y, s=1.4, c='blue', marker='o')
        #ax.scatter(next_x, next_y, s=1.2, c='#87CEFA', marker='o')

    # for i in range(p_array.shape[0]):
    #     for j in range(p_array.shape[1]):
    #         # Scaling the coordinates
    #         scaled_x = int(j * img_width / p_array.shape[1])
    #         scaled_y = int(i * img_height / p_array.shape[0])

    #         # Plotting the value
    #         ax.text(scaled_x, scaled_y, str(p_array[i, j]), color='blue', fontsize=8)
    if input_number:
        text_to_display = ""
        if img_filepath:
            text_to_display += f"img_filepath: {img_filepath}\n"
        for k, v in input_number.items():
            text_to_display += k + ':' + str(v) + '\n'

        ax.text(img_width * 0.9, img_height * 0.1, text_to_display, color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    # plt.show()

    plt.savefig(check_training_img_path+fileName, bbox_inches='tight')
    plt.close()
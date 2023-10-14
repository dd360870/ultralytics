def target_grid(target_x, target_y, stride):
    grid_x = int(target_x / stride)
    grid_y = int(target_y / stride)
    offset_x = (target_x % stride)
    offset_y = (target_y % stride)
    return grid_x, grid_y, offset_x, offset_y
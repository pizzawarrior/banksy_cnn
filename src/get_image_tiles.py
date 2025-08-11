
def make_img_tiles(img, h, w, overlap):
    '''
    take an image, and break it into as many tiles as possible
    using the provided params.
    overlap must be a percent of type float.
    NOTE: ALL TILES MUST BE THE SAME EXACT SIZE, NO EXCEPTIONS.
    '''

    def _verify_tile_size(tile):
        if tile.shape[1] != w and tile.shape[0] != h:
            tile = img[-h:, -w:]
        elif tile.shape[0] != h:
            tile = img[-h:, j:j + w]
        elif tile.shape[1] != w:
            tile = img[i:i + h, -w:]

        # extra validation
        if tile.shape == (h, w):
            return tile
        return None

    tiles = []
    orig_height, orig_width = img.shape

    if h > orig_height or w > orig_width:
        print(f'{"*"*50}\nRequested tile dims of {h}, {w} greater than image dims \
            of {img.shape}, please update the requested tile dims.\n{"*"*50}')
        return

    assert isinstance(overlap, float), 'Error: Overlap value must be float from 0-1.'
    offset = 1 - overlap
    overlap_h, overlap_w = int(offset * h), int(offset * w)

    for i in range(0, orig_height, h):
        for j in range(0, orig_width, w):
            tile = img[i:i + h, j:j + w]
            tiles.append(_verify_tile_size(tile))
            tile_overlap = img[i + overlap_h:i + overlap_h + h, j + overlap_w:j + overlap_w + w]
            tiles.append(_verify_tile_size(tile_overlap))

    return tiles

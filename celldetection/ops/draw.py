import torch


def draw_contours_(canvas, contours, close=True):
    """Draw contours.

    Draw ``contours`` on ``canvas``.

    Note:
        This is an inplace operation.

    Args:
        canvas: Tensor[h, w].
        contours: Contours in (x, y) format. Tensor[num_contours, num_points, 2].
        close: Whether to close contours. This is necessary if the last point of a contour is not equal to the first.

    """
    if close:
        contours = torch.cat((contours, contours[..., :1, :]), -2)
    diff = torch.diff(contours, axis=1)
    sign, diff = torch.sign(diff), torch.abs(diff)
    err = diff[..., 0] - diff[..., 1]
    x, y = contours[:, :-1, 0] + 0, contours[:, :-1, 1] + 0  # start point
    x_, y_ = contours[:, 1:, 0], contours[:, 1:, 1]  # end point
    labels = torch.broadcast_to(torch.arange(1, 1 + len(contours), device=canvas.device).to(canvas.dtype)[:, None],
                                x.shape)
    m = torch.ones(x.shape, dtype=torch.bool, device=canvas.device)
    while True:
        canvas[y[m], x[m]] = labels[m]
        # m: Select lines that are not finished; m_: Remove contours that are finished
        m = m & (((x != x_) | (y != y_)) & (x >= 0) & (y >= 0) & (x < canvas.shape[1]) & (y < canvas.shape[0]))
        m_ = torch.any(m, axis=-1)
        m = m[m_]
        if len(m) <= 0:
            break
        x, y, x_, y_, err, diff, sign, labels = (i[m_] for i in (x, y, x_, y_, err, diff, sign, labels))
        err_ = 2 * err
        sel = err_ > -diff[..., 1]
        err[sel] -= diff[sel][..., 1]  # Note: torch cannot handle arr[mask, index] properly; not equivalent to numpy
        x[sel] += sign[sel][..., 0]
        sel = err_ < diff[..., 0]
        err[sel] += diff[sel][..., 0]
        y[sel] += sign[sel][..., 1]

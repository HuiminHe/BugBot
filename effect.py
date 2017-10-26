from rendering import Line, LineWidth

def draw_lines(robots, radius):
    geoms = []
    locs = [r.loc() for r in robots]
    for i, r in enumerate(robots):
        for j, r_ in enumerate(robots[:i]):
            geom = Line((locs[i][0], locs[i][1]), (locs[j][0], locs[j][1]))
            geom.set_color_rgba([0, 1, 1, 1])
            geom.add_attr(LineWidth(3))
            geoms.append(geom)
    return geoms
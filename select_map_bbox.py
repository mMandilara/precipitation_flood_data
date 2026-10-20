import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

gdf = gpd.read_file('europe_countries.geojson')

fig, ax = plt.subplots(1, figsize=(10,7))
gdf.plot(facecolor='mediumseagreen', edgecolor='white', linewidth=0.3, ax=ax)
ax.set_xlim(-25, 45)
ax.set_ylim(34, 72)
ax.set_title('Select your area of interest', fontsize=12)
ax.text(
        0.25, 0.97, 'Draw a polygon to select the area of interest.\nPress Q to finish, Esc to reset.', transform=ax.transAxes, ha='center', va='top', fontsize=10, color='white',
        bbox=dict(boxstyle='round', facecolor='#DECD7A', alpha=0.85)
    )

polygon_coords = []
warning_text = None

def show_warning(msg):
    global warning_text
    clear_warning()
    warning_text = ax.text(
        0.5, -0.05, msg, transform=ax.transAxes,
        ha='center', va='top', fontsize=11, color='white',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.85)
    )
    fig.canvas.draw_idle()

def clear_warning():
    global warning_text
    if warning_text is not None:
        warning_text.remove()
        warning_text = None

def on_select(verts):
    global polygon_coords
    # needs to have 4 corners, sorry
    if len(verts)!=4:
        show_warning(f'Polygon must have 4 corners. You drew {len(verts)} corners. Pleaset try again, ty.')
        polygon_coords = []
        selector.clear()
        return
    clear_warning()
    polygon_coords = verts
    print(f'Polygon corners (lon, lat): {verts}')
    fig.canvas.draw_idle()

def close_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

def coords_hover(event):
    if event.inaxes:
        lon, lat = event.xdata, event.ydata
        ax.set_title(f'Lon: {lon:.2f}, Lat: {lat:.2f}', fontsize=10)
        plt.draw()

def main():
    selector = PolygonSelector(
        ax, useblit=True,
        handle_props=dict(color='blue', linestyle='-', linewidth=2, alpha=0.5),
        onselect=on_select
    )

    fig.canvas.mpl_connect('key_press_event', close_figure)
    fig.canvas.mpl_connect('motion_notify_event', coords_hover)

    plt.show()

    coords_int = ['%.2f' % coord for point in polygon_coords for coord in point]

    lons = [p[0] for p in polygon_coords]
    lats = [p[1] for p in polygon_coords]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    print(f'Selected bounding box: Lon min: {lon_min}, Lat min: {lat_min}, Lon max: {lon_max}, Lat max: {lat_max}')
    return lon_min, lat_min, lon_max, lat_max

lon_min, lat_min, lon_max, lat_max = main()
# if __name__ == '__main__':


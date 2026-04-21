import math
import os
import random

import matplotlib.pyplot as plt

try:
    import folium
except ImportError:
    folium = None


OUTPUT_DIR = "output"


def get_cities():
    """Return a default set of real Indian cities with GPS coordinates."""
    return {
        "Chennai": (13.0827, 80.2707),
        "Bangalore": (12.9716, 77.5946),
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Kolkata": (22.5726, 88.3639),
        "Hyderabad": (17.3850, 78.4867),
        "Pune": (18.5204, 73.8567),
        "Jaipur": (26.9124, 75.7873),
    }


def haversine_km(point_a, point_b):
    """Return real-world distance in km using the Haversine formula."""
    earth_radius_km = 6371

    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return earth_radius_km * c


def build_distance_matrix(cities):
    """Build a table of distances between every pair of cities."""
    names = list(cities.keys())
    matrix = {}

    for city_a in names:
        for city_b in names:
            matrix[(city_a, city_b)] = haversine_km(cities[city_a], cities[city_b])

    return matrix, names


def route_distance(route, matrix):
    """Calculate total distance including the return to the start city."""
    total = 0

    for index in range(len(route)):
        from_city = route[index]
        to_city = route[(index + 1) % len(route)]
        total += matrix[(from_city, to_city)]

    return total


def nearest_neighbour(matrix, names):
    """Build a greedy route by repeatedly visiting the nearest unvisited city."""
    unvisited = names.copy()
    route = [unvisited.pop(0)]

    while unvisited:
        current_city = route[-1]
        next_city = min(unvisited, key=lambda city: matrix[(current_city, city)])
        route.append(next_city)
        unvisited.remove(next_city)

    return route


def two_opt(route, matrix):
    """Improve the route by reversing route segments when they shorten the path."""
    best_route = route.copy()
    improved = True

    while improved:
        improved = False

        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                if i == 0 and j == len(best_route) - 1:
                    continue

                a, b = best_route[i], best_route[i + 1]
                c, d = best_route[j], best_route[(j + 1) % len(best_route)]

                current_distance = matrix[(a, b)] + matrix[(c, d)]
                swapped_distance = matrix[(a, c)] + matrix[(b, d)]

                if swapped_distance < current_distance - 1e-9:
                    best_route[i + 1 : j + 1] = reversed(best_route[i + 1 : j + 1])
                    improved = True

    return best_route


def format_route(route):
    return " -> ".join(route + [route[0]])


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_graph(cities, random_route, optimized_route, random_distance, optimized_distance):
    """Save a side-by-side graph comparing the random and optimized routes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("TSP Route Optimizer", fontsize=18, fontweight="bold", color="white")

    def draw(ax, route, color, title, distance, dashed):
        ax.set_facecolor("#1a1a2e")
        coordinates = [cities[city] for city in route] + [cities[route[0]]]
        longitudes = [point[1] for point in coordinates]
        latitudes = [point[0] for point in coordinates]
        linestyle = "--" if dashed else "-"

        ax.plot(longitudes, latitudes, color=color, linestyle=linestyle, linewidth=2.2)

        for index in range(len(route)):
            from_lat, from_lon = cities[route[index]]
            to_lat, to_lon = cities[route[(index + 1) % len(route)]]
            ax.annotate(
                "",
                xy=(to_lon, to_lat),
                xytext=(from_lon, from_lat),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            )

        for city, (latitude, longitude) in cities.items():
            ax.scatter(
                longitude,
                latitude,
                color="white",
                s=100,
                zorder=5,
                edgecolors=color,
                linewidths=1.4,
            )
            ax.annotate(
                city,
                (longitude, latitude),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        ax.set_title(
            f"{title}\nDistance: {distance:.1f} km\n{format_route(route)}",
            fontsize=10,
            color="white",
            pad=10,
        )
        ax.set_xlabel("Longitude", color="#bbbbbb")
        ax.set_ylabel("Latitude", color="#bbbbbb")
        ax.tick_params(colors="#bbbbbb")
        ax.grid(True, alpha=0.2, color="white")

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    draw(ax1, random_route, "#e74c3c", "Random Route", random_distance, True)
    draw(ax2, optimized_route, "#00d4aa", "Optimized Route", optimized_distance, False)

    savings = (
        ((random_distance - optimized_distance) / random_distance) * 100
        if random_distance
        else 0
    )
    fig.text(
        0.5,
        0.02,
        f"Improvement: {savings:.1f}% shorter | Saved: {random_distance - optimized_distance:.1f} km",
        ha="center",
        fontsize=12,
        color="#00d4aa",
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    output_path = os.path.join(OUTPUT_DIR, "graph.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"Graph saved to {output_path}")
    plt.close(fig)
    return output_path


def plot_real_map(cities, random_route, optimized_route, random_distance, optimized_distance):
    """Create an interactive real-world map using OpenStreetMap tiles."""
    if folium is None:
        print("\nReal map skipped because folium is not installed.")
        print("Install it with: pip install folium")
        return None

    average_latitude = sum(lat for lat, _ in cities.values()) / len(cities)
    average_longitude = sum(lon for _, lon in cities.values()) / len(cities)

    route_map = folium.Map(
        location=[average_latitude, average_longitude],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    random_coordinates = [cities[city] for city in random_route] + [cities[random_route[0]]]
    optimized_coordinates = [cities[city] for city in optimized_route] + [cities[optimized_route[0]]]

    folium.PolyLine(
        random_coordinates,
        color="red",
        weight=3,
        opacity=0.7,
        dash_array="8 10",
        tooltip=f"Random route: {random_distance:.1f} km",
    ).add_to(route_map)

    folium.PolyLine(
        optimized_coordinates,
        color="#00b894",
        weight=4,
        opacity=0.9,
        tooltip=f"Optimized route: {optimized_distance:.1f} km",
    ).add_to(route_map)

    for stop_number, city in enumerate(optimized_route, start=1):
        latitude, longitude = cities[city]
        popup_html = (
            f"<b>{city}</b><br>"
            f"Stop #{stop_number} in optimized route<br>"
            f"Lat: {latitude:.4f} | Lon: {longitude:.4f}"
        )

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=7,
            color="#1a1a2e",
            weight=2,
            fill=True,
            fill_color="#00d4aa",
            fill_opacity=0.95,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{stop_number}. {city}",
        ).add_to(route_map)

    savings = (
        ((random_distance - optimized_distance) / random_distance) * 100
        if random_distance
        else 0
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 1000;
        background: white;
        padding: 14px 18px;
        border: 1px solid #cfcfcf;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.12);
        font-family: Arial, sans-serif;
        font-size: 13px;
    ">
        <b>TSP Route Optimizer</b><br>
        <span style="color:red;">Random route:</span> {random_distance:.1f} km<br>
        <span style="color:#00b894;">Optimized route:</span> {optimized_distance:.1f} km<br>
        <span style="color:darkgreen;"><b>Improvement:</b> {savings:.1f}% shorter</span><br>
        <span style="color:#555555;">Route: {format_route(optimized_route)}</span>
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(legend_html))
    route_map.fit_bounds(list(cities.values()))

    output_path = os.path.join(OUTPUT_DIR, "map.html")
    route_map.save(output_path)
    print(f"Real map saved to {output_path}")
    return output_path


def main():
    print("=" * 50)
    print("TSP ROUTE OPTIMIZER")
    print("=" * 50)

    cities = get_cities()
    city_names = list(cities.keys())
    print(f"\nCities loaded: {city_names}")

    distance_matrix, city_names = build_distance_matrix(cities)
    print("\nDistance matrix built.")

    random_route = city_names.copy()
    random.shuffle(random_route)
    random_distance = route_distance(random_route, distance_matrix)
    print(f"\nRandom route:\n  {format_route(random_route)}")
    print(f"Distance: {random_distance:.1f} km")

    nearest_route = nearest_neighbour(distance_matrix, city_names)
    optimized_route = two_opt(nearest_route, distance_matrix)
    optimized_distance = route_distance(optimized_route, distance_matrix)
    print(f"\nOptimized route:\n  {format_route(optimized_route)}")
    print(f"Distance: {optimized_distance:.1f} km")

    savings = (
        ((random_distance - optimized_distance) / random_distance) * 100
        if random_distance
        else 0
    )
    print(f"\nImprovement: {savings:.1f}% shorter")
    print(f"Saved: {random_distance - optimized_distance:.1f} km")

    ensure_output_dir()
    print("\nGenerating graph...")
    plot_graph(cities, random_route, optimized_route, random_distance, optimized_distance)

    print("\nGenerating real-world map...")
    plot_real_map(cities, random_route, optimized_route, random_distance, optimized_distance)

    print("\nAll done. Check the output folder.")
    print("output/graph.png")
    print("output/map.html")


if __name__ == "__main__":
    main()

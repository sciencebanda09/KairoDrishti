// OpenStreetMap tiles
const map = L.map('map').setView([26.9124, 75.7873], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
}).addTo(map);

/**
 * Render one or more markers on the map.
 *
 * @param {Array<{lat: number, lng: number, title?: string, image?: string}>} locations
 *   List of locations. Each entry needs lat & lng. Optionally pass:
 *   - image: URL of an image. It becomes the small marker icon, and clicking
 *     the marker shows the image in a larger popup.
 *   - title: text shown alongside the image in the popup.
 */
function showMarkers(locations) {
    locations.forEach((loc) => {
        const icon = loc.image
            ? L.icon({
                  iconUrl: loc.image,
                  iconSize: [32, 32],
                  iconAnchor: [16, 16],
                  popupAnchor: [0, -16]
              })
            : new L.Icon.Default();

        const marker = L.marker([loc.lat, loc.lng], { icon }).addTo(map);

        // Build popup content
        let content = '';
        if (loc.title) content += `<strong>${loc.title}</strong>`;
        if (loc.image) content += `<img src="${loc.image}" alt="${loc.title || 'image'}">`;

        if (content) marker.bindPopup(content);
    });
}

// Example usage:
showMarkers([
    {
        lat: 26.9124,
        lng: 75.7873,
        title: 'Hawa Mahal',
        image: 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Hawa_Mahal_view_from_the_street_in_Jaipur.jpg/320px-Hawa_Mahal_view_from_the_street_in_Jaipur.jpg'
    },
    {
        lat: 26.9256,
        lng: 75.8073,
        title: 'Jaipur City Palace',
        image: 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/City_Palace_Jaipur.jpg/320px-City_Palace_Jaipur.jpg'
    },
    {
        lat: 26.9144,
        lng: 75.8218,
        title: 'Jantar Mantar'
    }
]);

// Fit map to show all markers
if (typeof L !== 'undefined') {
    const group = L.featureGroup(map._layers);
    if (group.getLayers().length) map.fitBounds(group.getBounds().pad(0.2));
}
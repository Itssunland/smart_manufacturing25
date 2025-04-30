# smart_manufacturing25

## 📱 Requirements: Phyphox App

This project relies on real-time sensor data from the [Phyphox app](https://phyphox.org/), available for **iOS** and **Android**.

### Setup Steps:
1. Install the **Phyphox** app on your phone.
2. Open the `Acceleration` experiment or your custom `.phyphox` experiment.
3. Enable **Remote Access** (click the Wi-Fi icon) and copy the given IP address (e.g. `http://192.168.1.100:8080`)
4. Add this URL to your `.env` file like so: "PHYPHOX_URL=http://<your-ip>:8080/get?"
5. Run the script to begin monitoring!

> The app must remain open and active while streaming data.

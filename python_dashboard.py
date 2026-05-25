from playwright.sync_api import sync_playwright

url = "http://127.0.0.1:5000"   # your dashboard URL

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={"width": 1600, "height": 900}
    )

    page.goto(url)

    # wait for charts/data loading
    page.wait_for_timeout(5000)

    # take full page screenshot
    page.screenshot(
        path="dashboard_full.png",
        full_page=True
    )

    browser.close()

print("Saved successfully!")
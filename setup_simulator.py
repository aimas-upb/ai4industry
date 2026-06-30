#!/usr/bin/env python3
"""
Setup script for the simulator.
- Resets all workstations (press emergency stop)
- Starts conveyor belts
- Places an item on the storage rack
"""

import argparse
import time
import httpx

BASE_URL = "https://ci.mines-stetienne.fr/simu"


def get_auth(group_num: int) -> tuple[str, str]:
    """Get username and password for the given group number."""
    return (f"simu{group_num}", f"simu{group_num}")


def reset_emergency_stops(group_num: int):
    """Call pressEmergencyStop on all workstations."""
    auth = get_auth(group_num)
    workstations = ["storageRack", "fillingWorkshop", "robotArm", "packagingWorkshop"]

    for ws in workstations:
        url = f"{BASE_URL}/{ws}/actions/pressEmergencyStop"
        print(f"Calling {ws}.pressEmergencyStop... ", end="", flush=True)
        try:
            response = httpx.post(url, auth=auth, timeout=10)
            if response.status_code >= 400:
                print(f"ERROR ({response.status_code}): {response.text}")
            else:
                print(f"OK ({response.status_code})")
        except Exception as e:
            print(f"ERROR: {e}")


def order_items(group_num: int):
    """Order items from providers."""
    auth = get_auth(group_num)
    orders = [
        ("cupProvider", "order", 25),
        ("cupProvider", "orderPackages", 5),
        ("dairyProductProvider", "order", 2),
    ]

    for provider, action, quantity in orders:
        url = f"{BASE_URL}/{provider}/actions/{action}"
        print(f"Calling {provider}.{action}({quantity})... ", end="", flush=True)
        try:
            response = httpx.post(url, auth=auth, json=quantity, timeout=10)
            if response.status_code >= 400:
                print(f"ERROR ({response.status_code}): {response.text}")
            else:
                print(f"OK ({response.status_code})")
        except Exception as e:
            print(f"ERROR: {e}")


def start_conveyor_belts(group_num: int):
    """Start conveyor belts at storageRack, fillingWorkshop, packagingWorkshop."""
    auth = get_auth(group_num)
    workstations = ["storageRack", "fillingWorkshop", "packagingWorkshop"]

    for ws_name in workstations:
        url = f"{BASE_URL}/{ws_name}/properties/conveyorSpeed"
        print(f"Starting {ws_name} conveyor belt... ", end="", flush=True)
        try:
            response = httpx.put(url, auth=auth, json=0.5, timeout=10)
            if response.status_code >= 400:
                print(f"ERROR ({response.status_code}): {response.text}")
            else:
                print(f"OK ({response.status_code})")
        except Exception as e:
            print(f"ERROR: {e}")


def wait_for_green_lights(group_num: int, timeout: int = 30):
    """Wait until all workstation stack lights are green."""
    auth = get_auth(group_num)
    workstations = ["storageRack", "fillingWorkshop", "packagingWorkshop"]

    print("Waiting for workstations to be ready (green lights)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        all_green = True
        for ws in workstations:
            url = f"{BASE_URL}/{ws}/all/properties"
            try:
                response = httpx.get(url, auth=auth, timeout=5)
                if response.status_code < 400:
                    data = response.json()
                    status = data.get("stackLightStatus", "").lower()
                    if status != "green":
                        all_green = False
                        break
            except Exception:
                all_green = False
                break

        if all_green:
            print("✓ All workstations ready (green lights)")
            return True

        print(".", end="", flush=True)
        time.sleep(1)

    print(f"\n✗ Timeout waiting for green lights after {timeout}s")
    return False


def place_item_on_storage(group_num: int):
    """Place an item on the storage rack conveyor belt."""
    auth = get_auth(group_num)
    url = f"{BASE_URL}/storageRack/actions/pickItem"

    print(f"Placing item on storage rack... ", end="", flush=True)
    try:
        response = httpx.post(url, auth=auth, json=[0, 0], timeout=10)
        if response.status_code >= 400:
            print(f"ERROR ({response.status_code}): {response.text}")
        else:
            print(f"OK ({response.status_code})")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup simulator for production run"
    )
    parser.add_argument(
        "group",
        type=int,
        choices=range(1, 11),
        help="Group number (1-10)",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only press emergency stop, skip conveyor and item setup",
    )

    args = parser.parse_args()
    group_num = args.group

    print(f"Setting up simulator for group {group_num}\n")

    print("=== Emergency Stop Reset ===")
    reset_emergency_stops(group_num)

    print("\n=== Ordering Items ===")
    order_items(group_num)

    if not args.reset_only:
        print("\n=== Waiting for Ready ===")
        if not wait_for_green_lights(group_num):
            print("Skipping belt/item setup due to timeout")
            return

        print("\n=== Starting Conveyor Belts ===")
        start_conveyor_belts(group_num)

        print("\n=== Placing Initial Item ===")
        place_item_on_storage(group_num)

    print("\n=== Setup Complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for the BT Planning Agent.

Sends a predefined carry goal to the agent and logs the results.
"""

import logging
import json
import httpx
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Goal specification
GOAL_PREDICATE = "!carry"
GOAL_SCHEMA = "!carry(RobotName, FromLocation, ToLocation)"
GOAL_INSTANCE = '!carry("APAS", "DX10_output", "XY10_input")'

# Agent endpoint
AGENT_URL = "http://localhost:8008/solve"


def main():
    """
    Send a test goal to the agent and print the results.

    The goal follows the predicate schema:
        !carry(RobotName, FromLocation, ToLocation)

    Concrete instance:
        !carry("APAS", "DX10_output", "XY10_input")

    This instructs the APAS robot arm to carry an item from the DX10 filling
    workstation's output area to the XY10 packaging workstation's input area.
    """
    logger.info(f"Testing BT Planning Agent")
    logger.info(f"Predicate schema: {GOAL_SCHEMA}")
    logger.info(f"Predicate instance: {GOAL_INSTANCE}")

    payload = {
        "goal": GOAL_INSTANCE,
        "execute": True  # Set to True to execute the generated BT
    }

    logger.info(f"Sending request to {AGENT_URL}")
    logger.debug(f"Payload: {json.dumps(payload)}")

    try:
        response = httpx.post(AGENT_URL, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        logger.info("Agent response received successfully")

        # Print results
        print("\n" + "="*80)
        print("AGENT TEST RESULTS")
        print("="*80)

        print(f"\nGoal: {result['goal']}")

        print(f"\n{'Capability Summary:'}")
        print("-" * 80)
        print(result['capability_summary'])

        print(f"\n{'Generated BehaviorTree Plan:'}")
        print("-" * 80)
        pprint(result['bt_plan'], width=80)

        if result['execution_result']:
            exec_result = result['execution_result']
            print(f"\n{'Execution Result:'}")
            print("-" * 80)
            print(f"Status: {exec_result['status']}")
            print(f"Ticks: {exec_result['ticks']}")

            if exec_result['status'] == 'SUCCESS':
                print(f"\n✓ Goal achieved!")
            elif exec_result['status'] == 'FAILURE':
                print(f"\n✗ Goal execution failed")
            else:
                print(f"\n⏱ Goal execution timed out")

            # Print execution trace
            if exec_result['trace']:
                print(f"\n{'Execution Trace:'}")
                print("-" * 80)
                for entry in exec_result['trace']:
                    status_symbol = "✓" if entry['status'] == 'SUCCESS' else "✗" if entry['status'] == 'FAILURE' else "→"
                    print(f"  [{entry['tick']:2d}] {status_symbol} {entry['node']:30s} [{entry['type']:20s}] {entry['status']}")
                    if entry['details']:
                        print(f"       └─ {entry['details']}")

        print("\n" + "="*80)
        logger.info("Test completed successfully")

    except httpx.ConnectError:
        logger.error(
            f"Could not connect to agent at {AGENT_URL}. "
            "Make sure the server is running: uvicorn src.main:app --reload"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Agent returned error {e.response.status_code}")
        logger.error(f"Response: {e.response.text}")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

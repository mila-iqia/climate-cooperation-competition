#!/bin/bash

# Set your API key
API_KEY="YOURKEY"

# Get the instances data
response=$(curl -s -X GET "https://cloud.lambdalabs.com/api/v1/instances" \
  -H 'accept: application/json' \
  -H "Authorization: Bearer $API_KEY")

# Extract the first instance ID from the response
instance_id=$(echo "$response" | grep -o '"id": "[^"]*"' | head -1 | cut -d'"' -f4)

# Check if we got an instance ID
if [ -z "$instance_id" ]; then
  echo "No instance ID found in the response"
  exit 1
fi

echo "Found instance ID: $instance_id"

# Call the terminate endpoint with the instance ID
terminate_response=$(curl -s -X POST "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate" \
  -H "accept: application/json" \
  -H "content-type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "{\"instance_ids\":[\"$instance_id\"]}")

echo "Termination response: $terminate_response"
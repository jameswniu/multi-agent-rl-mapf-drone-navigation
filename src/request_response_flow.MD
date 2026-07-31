# Request and response flow

## 1. Client sends a request

`POST /predict` with one observation, a row per drone:

```json
{ "state": [[0, 0, 19, 19, 200, 1, 1, 1, 0]] }
```

Each row is `[x, y, goal_x, goal_y, steps_remaining, blocked_up, blocked_down,
blocked_left, blocked_right]`. The number of rows must equal `num_drones` and
each row must have exactly nine values; either mismatch returns `422` before the
request reaches the policy.

## 2. Middleware intercepts

`add_metrics` records the start time, then after the endpoint returns it
increments `api_requests_total` and observes `api_request_latency_seconds`,
both labelled by method and endpoint, and logs a line such as:

```
2026-07-31 12:34:56 - api.app - INFO - POST /predict completed in 0.032s
```

## 3. The endpoint runs

The body is validated, then `PPOAgent.predict` returns one greedy action per
drone. The response carries both the indices and their names:

```json
{ "actions": [4], "action_names": ["right"] }
```

Indices are what `DroneEnv.step()` accepts; names are what a human reads in a
log. A failure inside the policy is wrapped as an `APIError` and rendered as
`{"error": "..."}` with status 500.

## 4. Response leaves the API

The middleware finishes timing, the metrics are already updated, and Prometheus
sees them on its next scrape.

## 5. Monitoring

Prometheus scrapes `GET /metrics`. The orchestrator polls `GET /healthz` and
restarts the container if it stops returning `{"status": "ok"}`.

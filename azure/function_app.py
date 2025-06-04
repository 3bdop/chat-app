import json
import logging

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="word_count", methods=["POST"])
def word_count(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Processing HTTP request for word_count.")

    # 1. Parse JSON body
    try:
        req_body = req.get_json()  # Expect {"values": [ { "recordId": "...", "data": { "text": "..." } }, ... ]}
        logging.error(f"reeeqqqq: {req_body}")
    except ValueError:
        return func.HttpResponse(
            'Invalid JSON payload. Expecting { "values": [ { "recordId": "...", "data": { "text": "..." } }, ... ] }',
            status_code=400,
        )

    # 2. Validate that "values" is present and is a list
    if "values" not in req_body or not isinstance(req_body["values"], list):
        return func.HttpResponse(
            'Payload must contain a top-level "values" array.', status_code=400
        )

    output_values = []
    for item in req_body["values"]:
        record_id = item.get("recordId")
        data = item.get("data", {})
        text = data.get("text", "") or ""
        # Compute word count: split on whitespace, ignore empty tokens
        count = len([w for w in text.strip().split() if w])
        output_values.append({"recordId": record_id, "data": {"wordCount": count}})

    response_payload = {"values": output_values}
    return func.HttpResponse(
        body=json.dumps(response_payload), status_code=200, mimetype="application/json"
    )
    #

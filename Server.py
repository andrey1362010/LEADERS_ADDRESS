from flask import Flask, request

app = Flask(__name__)

def process_address(name):
    return [
        (123, "Внуково"),
        (654, "Санкт-Петербург")
    ]

@app.route('/', methods=['GET'])
def process():
    args = request.args
    if "query" not in args: return "query param is required", 500
    query = args.get("query")
    if len(query) < 4: return "query param is too short", 500

    result = process_address(query)

    return {
        "success": True,
        "query:": {
            "address": query
        },
        "result": [
            {"target_building_id": building, "target_address": address} for building, address in result
        ]
    }

if __name__ == '__main__':
    app.run()
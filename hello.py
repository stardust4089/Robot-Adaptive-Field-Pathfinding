from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Grid Construction</title>
    <style>
        .grid {
            display: grid;
            grid-template-columns: repeat(165, 1fr);
            grid-template-rows: repeat(80, 1fr);
            gap: 1px;
            width: 1650px;
            height: 800px;
            background-color: #eee;
        }

        .square {
            background-color: gray;
        }

        .square.clicked {
            background-color: blue;
        }

        .square.obstacle {
            background-color: red;
        }
    </style>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script>
        $(document).ready(function() {
            $(".square").click(function() {
                var x = $(this).data("x");
                var y = $(this).data("y");

                $.post("/add_point", {"x": x, "y": y});

                $(this).toggleClass("clicked");
            });

            $("#construct-shape").click(function() {
                var points = $(".clicked");
                if (points.length === 4) {
                    var shapePoints = [];
                    points.each(function() {
                        var x = $(this).data("x");
                        var y = $(this).data("y");
                        shapePoints.push([x, y]);
                    });

                    $.post("/construct_shape", {"points": JSON.stringify(shapePoints)});

                    $(".square").removeClass("clicked");
                    $(".square").addClass("obstacle");
                } else {
                    alert("Please select 4 points to construct the shape.");
                }
            });
        });
    </script>
</head>
<body>
    <div class="grid">
        {% for y in range(80) %}
            {% for x in range(165) %}
                <div class="square" data-x="{{ x }}" data-y="{{ y }}"></div>
            {% endfor %}
        {% endfor %}
    </div>
    <button id="construct-shape">Construct Shape</button>
</body>
</html>"""
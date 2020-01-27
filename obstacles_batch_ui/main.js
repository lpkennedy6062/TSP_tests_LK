window.tspCities = []
window.tspObstacles = []
window.tourEdges = []
window.tourEdgeTimes = []
window.visitedCities = []
window.activeVertices = []
window.currentVertex = null

window.problemIndex = 0
window.tourComplete = false
window.setCompleted = false

window.cHeight = 500
window.cWidth = 500
window.c = document.getElementById("mainCanvas")
window.ctx = c.getContext("2d")

var _array = new Uint32Array(1)
window.crypto.getRandomValues(_array)
window.clientId = _array[0]


var MAX_DISTANCE = 16


// Utility functions

function sameCoord(xy1, xy2)
{
    return xy1[0] === xy2[0] && xy1[1] === xy2[1]
}

function inArray(xy1, a)
{
    for (var i = 0; i < a.length; i++)
        if (sameCoord(xy1, a[i]))
            return true
    return false
}

function isSubset(a, b)
{
    for (var i = 0; i < a.length; i++)
        if (!inArray(a[i], b))
            return false
    return true
}

function checkComplete()
{
    if (tourEdges.length < 2)
        return false
    // Is the tour closed?
    if (!sameCoord(tourEdges[0][0], tourEdges[tourEdges.length - 1][1]))
        return false
    // Are all cities in the tour?
    if (!isSubset(tspCities, tourEdges.flat()))
        return false
    return true
}

function distance(xy1, xy2)
{
    var dx = xy2[0] - xy1[0]
    var dy = xy2[1] - xy1[1]
    return Math.sqrt((dx * dx) + (dy * dy))
}


var CURRENT = "#0f0"
var ACTIVE = "#00f"
var INACTIVE = "#f00"
var NONCITY = "#999"

function drawVertex(x, y, color)
{
    var r = 5
    ctx.beginPath()
    ctx.arc(x, y, r, 0, 2 * Math.PI, false)
    ctx.fillStyle = color
    ctx.fill()
}

function drawEdge(xy1, xy2, color, thickness)
{
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.strokeStyle = color
    ctx.lineWidth = thickness
    ctx.stroke()
}

function drawCities()
{
    tspCities.forEach(function (xy) {
        drawVertex(xy[0], xy[1], INACTIVE)
    })
}

function drawObstacles()
{
    tspObstacles.forEach(function (ab) {
        xy1 = ab[0]
        xy2 = ab[1]
        drawEdge(xy1, xy2, "#000", 3)
    })
}

function drawTour()
{
    tourEdges.forEach(function (ab) {
        xy1 = ab[0]
        xy2 = ab[1]
        drawEdge(xy1, xy2, "#00f", 2)
    })
}

function drawActive()
{
    activeVertices.forEach(function (xy) {
        drawVertex(xy[0], xy[1], (inArray(xy, tspCities)) ? ACTIVE : NONCITY)
    })
    if (currentVertex !== null) {
        drawVertex(currentVertex[0], currentVertex[1], CURRENT)
    }
}

function clearDrawing()
{
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, cHeight, cWidth)
}

function resize()
{
    c.height = cHeight * 2
    c.width = cWidth * 2
    window.ctx = c.getContext("2d")
    ctx.scale(2, 2)
}

function redraw()
{
    clearDrawing()
    resize()
    drawCities()
    drawObstacles()
    drawTour()
    drawActive()
}


// API functions

function clear()
{
    window.tspCities = []
    window.tspObstacles = []
    window.tourEdges = []
    window.tourEdgeTimes = []
    window.activeVertices = []
    window.currentVertex = null
}

function nextProblem()
{
    var xhr = new XMLHttpRequest()
    xhr.open("GET", "/api/" + window.problemIndex + "/cities", true)
    xhr.send(null)
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            clear()
            if (xhr.status == 200) {
                var response = JSON.parse(xhr.responseText)
                window.tspCities = response.cities
                window.tspObstacles = response.obstacles
                window.activeVertices = response.cities
                window.cHeight = response.height
                window.cWidth = response.width
                redraw()
                document.getElementById("bottomBar").innerText = "Problem " + (window.problemIndex + 1)
            } else {
                window.setCompleted = true
                window.alert("No more problems!")
            }
        }
    }
}

function getVisgraph()
{
    if (inArray(currentVertex, tspCities) && tourEdges.length > 0 && !sameCoord(currentVertex, tourEdges[0][0]))
        visitedCities.push(currentVertex)
    var xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/" + window.problemIndex + "/visgraph", true)
    xhr.send("data=" + encodeURIComponent(JSON.stringify(currentVertex)))
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            if (xhr.status == 200) {
                window.activeVertices = JSON.parse(xhr.responseText).filter(function(v) {
                    return !inArray(v, visitedCities)
                })
                redraw()
            } else {
                window.alert("Error encountered! Is the server still running?")
            }
        }
    }
}

function recordTour()
{
    var xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/" + window.problemIndex + "/tour", true)
    xhr.send("data=" + encodeURIComponent(JSON.stringify([tourEdges, tourEdgeTimes])))
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            window.problemIndex++
            nextProblem()
        }
    }
}


// UI functions

function getXY(ev)
{
    var rect = ev.target.getBoundingClientRect()
    var x = ev.clientX - rect.left
    var y = ev.clientY - rect.top
    x = x * cWidth / c.width
    y = y * cHeight / c.height
    return [x, y]
}

function restartTour(ev)
{
    if (!confirm("Clear existing tour?"))
        return
    window.tourEdges = []
    window.tourEdgeTimes = []
    window.visitedCities = []
    window.currentVertex = null
    window.activeVertices = tspCities
    window.tourComplete = false
    redraw()
}

function undoTour(ev)
{
    if (tourEdges.length === 0)
        return
    while (inArray(currentVertex, visitedCities))
        visitedCities.pop()
    window.currentVertex = tourEdges.pop()[0]
    tourEdgeTimes.pop()
    getVisgraph()
    window.tourComplete = false
}

function sendTour(ev)
{
    if (!tourComplete)
    {
        alert("Tour incomplete!")
        return
    }
    recordTour()
    window.tourComplete = false
}

function buildTour(ev)
{
    if (tourComplete)
    {
        restartTour()
    }
    var xy = getXY(ev)
    var min = Infinity
    var nextVertex = null
    for (var i = 0; i < activeVertices.length; i++)
    {
        var dist = distance(xy, activeVertices[i])
        if (!inArray(activeVertices[i], tspCities))
            dist *= 2
        if (dist < min)
        {
            min = dist
            nextVertex = activeVertices[i]
        }
    }
    if (min > MAX_DISTANCE)
        nextVertex = null
    if (nextVertex === null)
    {
        // alert("No active vertices!")
        return
    }
    if (window.currentVertex === null)
    {
        window.tourEdgeTimes.push(new Date().getTime())
        window.currentVertex = nextVertex
        getVisgraph()
        return
    }
    tourEdges.push([currentVertex, nextVertex])
    tourEdgeTimes.push(new Date().getTime())
    if (checkComplete())
    {
        window.tourComplete = true
        window.activeVertices = []
        window.currentVertex = nextVertex
        redraw()
        setTimeout(function(){ alert("Tour complete!"); }, 100);
    }
    else
    {
        window.currentVertex = nextVertex
        getVisgraph()
    }
}

function cancelReload(ev)
{
    return setCompleted
}

window.c.onclick = buildTour
window.onbeforeunload = cancelReload
document.getElementById("clearButton").onclick = restartTour
document.getElementById("undoButton").onclick = undoTour
document.getElementById("submitButton").onclick = sendTour
nextProblem()

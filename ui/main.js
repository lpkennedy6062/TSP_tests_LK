// Global variables - only one set of cities and one tour can be displayed at one time

window.tspCities = []
window.tspTour = []
window.currentCity = -1

window.c = document.getElementById("mainCanvas")
window.ctx = c.getContext("2d")

var _array = new Uint32Array(1)
window.crypto.getRandomValues(_array)
window.clientId = _array[0]


// Utility functions

function distance(xy1, xy2)
{
    var dx = xy2[0] - xy1[0]
    var dy = xy2[1] - xy1[1]
    return Math.sqrt((dx * dx) + (dy * dy))
}

function drawCity(x, y, selected)
{
    var r = 5
    ctx.beginPath()
    ctx.arc(x, y, r, 0, 2 * Math.PI, false)
    ctx.fillStyle = (selected) ? "#00f" : "#f00"
    ctx.fill()
}

function drawEdge(a, b)
{
    var x1 = tspCities[a][0]
    var y1 = tspCities[a][1]
    var x2 = tspCities[b][0]
    var y2 = tspCities[b][1]

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.strokeStyle = "#00f"
    ctx.lineWidth = 2
    ctx.stroke()
}

function drawCities()
{
    tspCities.forEach(function (xy) {
        drawCity(xy[0], xy[1])
    })
    if (currentCity >= 0)
    {
        drawCity(tspCities[currentCity][0], tspCities[currentCity][1], true)
    }
}

function drawEdges()
{
    var a = 0, b = 1
    for (; b < tspTour.length; a++, b++)
    {
        drawEdge(tspTour[a], tspTour[b])
    }
    if (tspTour.length > 0 && currentCity < 0) {
        drawEdge(tspTour[a], tspTour[0])
    }
}

function clearDrawing()
{
    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, 500, 500)
}

function redraw()
{
    clearDrawing()
    drawEdges()
    drawCities()
}


// API functions

function pushCities()
{
    var xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/" + clientId + "/push", true)
    xhr.send("data=" + encodeURIComponent(JSON.stringify(tspCities)))
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            redraw()
        }
    }
}

function getConcordeSolution()
{
    if (tspCities.length < 4) { return }
    var xhr = new XMLHttpRequest()
    xhr.open("GET", "/api/" + clientId + "/solve/concorde", true)
    xhr.send(null)
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            window.tspTour = JSON.parse(xhr.responseText)
            redraw()
        }
    }
}

function getChristofidesSolution()
{
    if (tspCities.length < 4) { return }
    var xhr = new XMLHttpRequest()
    xhr.open("GET", "/api/" + clientId + "/solve/christofides", true)
    xhr.send(null)
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            window.tspTour = JSON.parse(xhr.responseText)
            redraw()
        }
    }
}

function getPyramidSolution()
{
    if (tspCities.length < 4) { return }
    var xhr = new XMLHttpRequest()
    xhr.open("GET", "/api/" + clientId + "/solve/pyramid", true)
    xhr.send(null)
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            window.tspTour = JSON.parse(xhr.responseText)
            redraw()
        }
    }
}

function getRandomCities(n)
{
    var xhr = new XMLHttpRequest()
    xhr.open("GET", "/api/" + clientId + "/generate/" + n, true)
    xhr.send(null)
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            window.tspCities = JSON.parse(xhr.responseText)
            redraw()
        }
    }
}

function getTourScore()
{
    if (tspCities.length < 4) { return }
    var xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/" + clientId + "/score", true)
    xhr.send("data=" + encodeURIComponent(JSON.stringify(tspTour)))
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            var r = JSON.parse(xhr.responseText)
            var score = r.score
            var optimal = r.optimal
            var loss = ((r.score + r.optimal) / (2.0 * r.optimal)) - 1.0
            alert("Distance Travelled: " + Math.round(score) + "\nOptimal Path: " + Math.round(optimal) + "\nError: " + Math.round(loss * 10000) / 10000)
        }
    }
}


// UI functions

function getXY(ev)
{
    var rect = ev.target.getBoundingClientRect()
    var x = ev.clientX - rect.left
    var y = ev.clientY - rect.top
    return [x, y]
}

function clear(ev)
{
    window.tspCities = []
    window.tspTour = []
    redraw()
}

function solveOptimal(ev)
{
    getConcordeSolution()
}

function solveChristofides(ev)
{
    getChristofidesSolution()
}

function solvePyramid(ev)
{
    getPyramidSolution()
}

function placeCity(ev)
{
    tspCities.push(getXY(ev))
    pushCities()
}

function randomCities(ev)
{
    clear()
    var n = prompt("How many cities?", "10")
    if (n) {
        getRandomCities(n)
    }
}

function restartTour(ev)
{
    window.tspTour = []
    redraw()
}

function scoreTour(ev)
{
    if (tspTour.length != tspCities.length)
    {
        alert("Tour incomplete!")
        return
    }
    getTourScore()
}

function buildTour()
{
    window.prevCity = -1
    return function(ev) {
        if (prevCity < 0 && tspTour.length > 0)
        {
            if (!confirm("Clear existing tour?"))
            {
                return
            }
            restartTour()
        }
        var xy = getXY(ev)
        var min = Infinity
        window.currentCity = -1
        for (var i = 0; i < tspCities.length; i++)
        {
            var dist = distance(xy, tspCities[i])
            if (dist < min)
            {
                min = dist
                window.currentCity = i
            }
        }
        if (tspTour.indexOf(currentCity) >= 0)
        {
            window.currentCity = prevCity
            return
        }
        tspTour.push(currentCity)
        if (tspTour.length < tspCities.length)
        {
            window.prevCity = currentCity
        }
        else
        {
            alert("Tour complete!")
            window.prevCity = -1
            window.currentCity = -1
        }
        redraw()
    }
}


// UI modalities

function changeMode()
{
    var s = document.getElementById("modeSelect")
    var t = document.getElementById("modeTools")
    if (s.value == "edit")
    {
        t.innerHTML = '<input type="button" value="Clear" id="clearButton" /> ' +
                      '<input type="button" value="Random" id="randomButton" />'
        document.getElementById("clearButton").onclick = clear
        document.getElementById("randomButton").onclick = randomCities
        window.c.onclick = placeCity
        window.currentCity = -1
    }
    else if (s.value == "solve")
    {
        t.innerHTML = '<input type="button" value="Restart" id="restartButton" /> ' +
                      '<input type="button" value="Score" id="scoreButton" /> | ' +
                      '<input type="button" value="Concorde" id="optimalButton" /> ' +
                      '<input type="button" value="Christofides" id="christofidesButton" /> ' +
                      '<input type="button" value="Pyramid" id="pyramidButton" />'
        document.getElementById("restartButton").onclick = restartTour
        document.getElementById("scoreButton").onclick = scoreTour
        document.getElementById("optimalButton").onclick = solveOptimal
        document.getElementById("christofidesButton").onclick = solveChristofides
        document.getElementById("pyramidButton").onclick = solvePyramid
        window.c.onclick = buildTour()
    }
    redraw()
}
document.getElementById("modeSelect").onchange = changeMode
changeMode()

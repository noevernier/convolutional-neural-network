let user 
let size = 28
let button


function setup(){
    createCanvas(500,500)

    user = createGraphics(size, size);
    user.background(0);
    user.noStroke();

    button = createButton('Save')
    button.position(240, 520)
    button.mousePressed(saveFigure);
}
function mouseDragged() {
    user.fill(255,255,255,180)
    user.ellipse(mouseX/(width/size), mouseY/(height/size), 2, 2);
  }

function draw(){
    background(0)
    image(user, 0, 0, 500, 500);
}

function saveFigure(){
    fig = []
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            val = user.get(i,j)[0]/255
            fig[size*j + i] = val 
        }
    }
    output = JSON.stringify(fig);
    saveJSON(output, "output")
}
function makePxl(x,y) {
    let pxlContainer = document.getElementById('pxlArray');
    let pxlBox = document.createElement('div');
    pxlBox.id = 'pxl_x-' + x + '_y-' + y;
    pxlContainer.appendChild(pxlBox);
    document.getElementById(pxlBox.id).style.width = '10px';
    document.getElementById(pxlBox.id).style.height = '10px';
    document.getElementById(pxlBox.id).style.backgroundColor = 'lightblue';
}
function updatePxlColor(x,y,color) {
    let pxlBox = document.getElementById('pxl_x-' + x + '_y-' + y);
    pxlBox.style.backgroundColor = 'rgba('+color*255+','+color*255+','+color*255+',1)';
}
function makeVal(x,y) {
    let valContainer = document.getElementById('valArray');
    let valBox = document.createElement('div');
    valBox.id = 'val_x-' + x + '_y-' + y;
    valContainer.appendChild(valBox);
    document.getElementById(valBox.id).style.display = 'block';
    document.getElementById(valBox.id).style.width = '8px';
    document.getElementById(valBox.id).style.height = '8px';
    document.getElementById(valBox.id).textContent = '0';
}
function updateValColor(x,y,color) {
    let valBox = document.getElementById('val_x-' + x + '_y-' + y);
    valBox.style.fontSize = '10px'
    valBox.textContent = color;
}
function makeOut(x) {
    let neurons = document.getElementById('neurons');
    let out = document.createElement('p');
    out.id = 'out-' + x;
    neurons.appendChild(out);
    document.getElementById(out.id).style.margin = "0px";
}
function updateOut(x,value) {
    let out = document.getElementById('out-' + x);
    out.innerHTML = "Neuron #"+x+"<br>"+value;
}
function newStream() {
    const currentTime = () => new Date().getFullYear() + '/' + (new Date().getMonth()+1) + '/' + new Date().getDate() + ' ' + new Date().getHours() + ':' + new Date().getMinutes() + ':' + new Date().getSeconds();
    let init = 0;
    let socket = io.connect('http://' + document.domain + ':' + location.port + '/',{transports: ["polling",'websocket']});

    setInterval(() => {
      socket.emit('current frame', {time: currentTime(), data: 'Give me new frame.'});
    }, 1/10 * 1000);

    socket.emit('current frame', {time: currentTime(), data: 'Give me new frame.'});

    socket.on('new frame', function(msg) {
	//console.log(msg.info)
	if (init == 0) {
	    document.getElementById('pxlArray').style.width = (msg.data.length*10) + 'px';
	    document.getElementById('pxlArray').style.height = (msg.data.length*10) + 'px';
	    document.getElementById('valArray').style.width = (msg.data.length*16) + 'px';
	    document.getElementById('valArray').style.height = (msg.data.length*8) + 'px';
	    for (var i = 0; i < msg.data.length; i++) {
		for (var j = 0; j < msg.data[i].length; j++) {
		    makePxl(i,j);
		    makeVal(i,j);
		}
	    }
	    for (var i = 0; i < msg.out.length; i++) {
		makeOut(i);
	    }
	    init = 1;
	}
	if (msg.data.length != 0) {
	    for (var i = 0; i < msg.data.length; i++) {
		for (var j = 0; j < msg.data[i].length; j++) {
                    var val = 0;
                    if (msg.data[i][j] == true) {
                        val = 1;
                    } else {
                        val = 0;
                    }
                    updatePxlColor(i,j,val);
		    updateValColor(i,j,val);
		}
	    }
	}
	if (msg.ans.length != 0) {
	    ans = document.getElementById('answer');
            ans.style.fontSize = '96px';
	    if (msg.ans == 0) {
		ans.innerHTML = "PacMan";
	    } else if (msg.ans == 1) {
		ans.innerHTML = "Okrąg";
	    } else if (msg.ans == 2) {
		ans.innerHTML = "Kwadrat";
	    } else if (msg.ans == 3) {
		ans.innerHTML = "Trójkąt";
	    }
            //ans.innerHTML = msg.ans;
	}
	if (msg.out.length != 0) {
	    for (var i = 0; i < msg.out.length; i++) {
		updateOut(i,msg.out[i]);
	    }
	    console.log(msg.out);
	    
	}
		
    });
    
}



newStream();

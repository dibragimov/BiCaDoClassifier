

function importData() {
  var img = document.getElementById("selected_image");
  let input = document.createElement('input');
  input.type = 'file';
  let byteFile = null
  input.onchange = _ => {
    // you can use this method to get file and perform respective operations
            let files = Array.from(input.files);
			const reader = new FileReader();
			reader.addEventListener('load', (event) => {
				byteFile = event.target.result;
				img.src = byteFile;
				// byteFile = byteFile.split('base64,')[1] // get only image data
				// console.log(byteFile.length);
				// console.log(byteFile);
				// byteFile = base64ToArrayBuffer(byteFile)
				// byteFile = Base64Binary.decodeArrayBuffer(byteFile);
				console.log(byteFile);
				send_image(byteFile);
				});
			reader.readAsDataURL(files[0]);
			// 
        };
  input.click();
  
}

function send_image(bytearray){
    var x = document.getElementById("replytext");
    // console.log(req)
    // x.innerHTML = req;
    // test
    var request = new XMLHttpRequest();

    // Open a new connection, using the GET request on the URL endpoint
    request.open("POST", "http://localhost:8000/predict"); 
    request.setRequestHeader("Content-Type", "application/json");
    request.onload = function() {
        // Begin accessing JSON data here
	var data = JSON.parse(this.response);
        //console.log(data)
        //console.log(request.status)
	if (request.status >= 200 && request.status < 400) {
            x.value = 'Class: '+data['class']+', Probability: '+data['probability'];
            //console.log(x.value)
	} else {
             msg = ' '
             if(data['message'] != null){ msg = data['message'] }
             x.value = 'Error, status: '+request.status + msg;
        }
    }
    var data = JSON.stringify({"image": [bytearray]}); 
    console.log(data);
    request.send(data);
}

function base64ToArrayBuffer(base64) {
    var binaryString = atob(base64);
    var bytes = new Uint8Array(binaryString.length);
    for (var i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

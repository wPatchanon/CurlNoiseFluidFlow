const canvas = document.querySelector("#glCanvas");
var gl;
var shaderProgram;
var camera = {};
var programInfo = {};
var then = 0;
var rotation = 0;
var config = {
    twice: true,
    rotate: false,
};
var particles = {
    num: 1 << 16,
    size: 1.0,
    speed: 0.004,
    perlin_ratio: 1.2,
    ramp_ratio: 1.5,
    force_multiply: 2.0,

};
var sphere = {
    render: true,
    radius: 2.5,
    pos_x: 0.0,
    pos_y: 3.0,
    pos_z: 0,
}
var sphere2 = {
    render: true,
    radius: 1.0,
    pos_x: 0,
    pos_y: -3.0,
    pos_z: 0,
}


// Show FPS
var stats = new Stats()
stats.showPanel(0)
document.body.appendChild(stats.dom)

// Show Slider
var gui = new dat.GUI();
gui.add(particles, 'speed', 0.001, 0.1);
gui.add(particles, 'size', 0.5, 3.0);
var pNumController = gui.add(particles, 'num', 1 << 12, 1 << 18);
gui.add(particles, 'ramp_ratio', 1.0, 25.0);
gui.add(particles, 'perlin_ratio', 1.0, 5.0);
gui.add(particles, 'force_multiply', 1.0, 5.0);
gui.add(config, 'rotate');
gui.add(config, 'twice');
gui.add(sphere, 'render');
var radiusController = gui.add(sphere, 'radius', 1.0, 5.0);
var posXController = gui.add(sphere, 'pos_x', -15.0, 15.0);
var posYController = gui.add(sphere, 'pos_y', -15.0, 15.0);
gui.add(sphere2, 'render');
var radiusController2 = gui.add(sphere2, 'radius', 1.0, 5.0);
var posXController2 = gui.add(sphere2, 'pos_x', -15.0, 15.0);
var posYController2 = gui.add(sphere2, 'pos_y', -15.0, 15.0);


pNumController.onChange(initParticles)
posXController.onChange(() => { return initSphere(sphere); })
posYController.onChange(() => { return initSphere(sphere); })
radiusController.onChange(() => { return initSphere(sphere); })
posXController2.onChange(() => { return initSphere(sphere2); })
posYController2.onChange(() => { return initSphere(sphere2); })
radiusController2.onChange(() => { return initSphere(sphere2); })

initShaders()
initParticles()
initSphere(sphere)
initSphere(sphere2)
initCamera()
// renderSphere()
requestAnimationFrame(renderLoop)
// renderParticles()

function renderLoop(now) {
    stats.begin();

    now *= 0.001;
    const dt = now - then;
    then = now;
    if (config.rotate) {
        rotation += dt;
        mat4.rotate(camera.modelViewMatrix,
            camera.modelViewMatrix,
            rotation * 0.0007,
            [0, 1, 0]);
    }

    renderParticles();
    if (sphere.render) renderSphere(sphere);
    if (config.twice && sphere2.render) renderSphere(sphere2);

    stats.end();
    requestAnimationFrame(renderLoop)
}

function initShaders() {
    gl = canvas.getContext("webgl");
    canvas.width = gl.canvas.clientWidth
    canvas.height = gl.canvas.clientHeight
    console.log(canvas.width, canvas.height)
    gl.viewport(0, 0, canvas.width, canvas.height);
    // console.log(gl.canvas.clientWidth, gl.canvas.clientHeight)
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.getExtension('OES_texture_float')
    gl.getExtension('WEBGL_color_buffer_float');

    var perlin_noise = `  
    vec3 mod289(vec3 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    vec4 mod289(vec4 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    vec4 permute(vec4 x) {
        return mod289(((x*34.0)+1.0)*x);
    }
    vec4 taylorInvSqrt(vec4 r)
    {
      return 1.79284291400159 - 0.85373472095314 * r;
    }
    float snoise(vec3 v)
    { 
      const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
      const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    // First corner
      vec3 i  = floor(v + dot(v, C.yyy) );
      vec3 x0 =   v - i + dot(i, C.xxx) ;
    // Other corners
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min( g.xyz, l.zxy );
      vec3 i2 = max( g.xyz, l.zxy );
      //   x0 = x0 - 0.0 + 0.0 * C.xxx;
      //   x1 = x0 - i1  + 1.0 * C.xxx;
      //   x2 = x0 - i2  + 2.0 * C.xxx;
      //   x3 = x0 - 1.0 + 3.0 * C.xxx;
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
      vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y
    // Permutations
      i = mod289(i); 
      vec4 p = permute( permute( permute( 
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
      float n_ = 0.142857142857; // 1.0/7.0
      vec3  ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
      vec4 x = x_ *ns.x + ns.yyyy;
      vec4 y = y_ *ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4( x.xy, y.xy );
      vec4 b1 = vec4( x.zw, y.zw );
      //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
      //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
      vec4 s0 = floor(b0)*2.0 + 1.0;
      vec4 s1 = floor(b1)*2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
      vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
      vec3 p0 = vec3(a0.xy,h.x);
      vec3 p1 = vec3(a0.zw,h.y);
      vec3 p2 = vec3(a1.xy,h.z);
      vec3 p3 = vec3(a1.zw,h.w);
    //Normalise gradients
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;
    // Mix final noise value
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                    dot(p2,x2), dot(p3,x3) ) );
    }
  `;


    const positionVS = `
        precision mediump float;

        attribute vec4 aPosition;

        void main(void) {
            gl_Position = aPosition;
        }
    `;

    const positionFS = `
        precision mediump float;

        uniform sampler2D uPositionTexture;
        uniform sampler2D uSpawnTexture;
        uniform vec2 uResolution;
        uniform float uSpeed;
        uniform vec4 uSphere1;
        uniform vec4 uSphere2;
        uniform vec4 uConf;
        uniform bool uTwice;
        
        ${perlin_noise}

        float noise1(vec3 v) {
            return snoise(v);
        }
        float noise2(vec3 v) {
            return snoise(vec3(v.y+45.345, v.z+12.344, v.x+60.342));
        }
        float noise3(vec3 v) {
            return snoise(vec3(v.z-22.545, v.x+23.654, v.y-54.342));
        }
        vec3 noise3d(vec3 v) {
            return vec3(noise1(v), noise2(v), noise3(v));
        }
        float ramp(float r) {
            if(r >= 1.0)
                return 1.0;
            else 
                return 1.875*r - 1.25*r*r*r + 0.375*r*r*r*r*r;
        }
        float findDistance1(vec3 v) {
            vec3 dist = v - uSphere1.xyz;
            return length(dist) - uSphere1.w;
        }
        float findDistance2(vec3 v) {
            vec3 dist = v - uSphere2.xyz;
            return length(dist) - uSphere2.w;
        }
        vec3 potential(vec3 v) {
            float r1 = findDistance1(v) / uConf.x;
            float rmp1 = ramp(r1);
            vec3 n1 = normalize(v-uSphere1.xyz);

            vec3 noise = noise3d(v / uConf.y);
            vec3 nForce = vec3(v.z, 0.0, -v.x);

            vec3 field = noise + uConf.z * nForce;

            if(uTwice) {
                float r2 = findDistance2(v) / uConf.x;
                float rmp2 = ramp(r2);
                float rmp = (r2 * rmp1 + r1 * rmp2) / (rmp1 + rmp2);
                vec3 n2 = normalize(v-uSphere2.xyz);
                return rmp * field + rmp2 * (1.0-rmp1) * dot(n1,field) * n1 + rmp1 * (1.0-rmp2) * dot(n2,field) * n2;
            }
            else
                return rmp1 * field + (1.0-rmp1) * dot(n1,field) * n1;
        }
        vec3 computeCurl(vec3 v) {
            const float e = 0.01;
            vec3 dx = vec3(e,0,0);
            vec3 dy = vec3(0,e,0);
            vec3 dz = vec3(0,0,e);

            float vx = potential(v+dy).z - potential(v-dy).z - potential(v+dz).y + potential(v-dz).y; 
            float vy = potential(v+dz).x - potential(v-dz).x - potential(v+dx).z + potential(v-dx).z; 
            float vz = potential(v+dx).y - potential(v-dx).y - potential(v+dy).x + potential(v-dy).x; 

            return vec3(vx,vy,vz)/(2.0*e);
        }

        void main() {
            vec2 texCoor = gl_FragCoord.xy / uResolution;
            vec4 data = texture2D(uPositionTexture,texCoor);
            float oldtime = data.a;
            float newtime = oldtime - 0.001;
            if(newtime <= 0.0) {
                vec4 newp = texture2D(uSpawnTexture, texCoor);
                gl_FragColor = newp;
            }
            else
                gl_FragColor = vec4(data.rgb + uSpeed * computeCurl(data.rgb) ,newtime);
        }
    `;

    const renderVS = `
        precision mediump float;

        attribute vec2 aTexCoor;

        uniform mat4 uModelViewMatrix;
        uniform mat4 uProjectionMatrix;
        uniform sampler2D uPositionTexture;
        uniform float uSize;

        varying vec4 vColor;

        void main(void) {
            vec4 pos = texture2D(uPositionTexture,aTexCoor);
            gl_PointSize = uSize;
            gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(pos.rgb,1.0);
            vColor = vec4(pos);
        }
    `;

    const renderFS = `
        precision mediump float;

        varying vec4 vColor;

        void main() {
            gl_FragColor = vColor;
        }
    `;

    const sphereVS = `
        precision mediump float;

        attribute vec4 aPosition;

        uniform mat4 uModelViewMatrix;
        uniform mat4 uProjectionMatrix;
        uniform mat4 uModMat;

        void main(void) {
            gl_Position = uProjectionMatrix * uModelViewMatrix * uModMat * aPosition;
        }
    `;

    const sphereFS = `
        precision mediump float;

        void main() {
            gl_FragColor = vec4(0.4,0.4,0.4,1.0);
        }
    `;

    positionShader = initShaderProgram(positionVS, positionFS);
    positionShaderInfo = {
        program: positionShader,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(positionShader, 'aPosition'),
        },
        uniformLocations: {
            positionTexture: gl.getUniformLocation(positionShader, 'uPositionTexture'),
            spawnTexture: gl.getUniformLocation(positionShader, 'uSpawnTexture'),
            resolution: gl.getUniformLocation(positionShader, 'uResolution'),
            speed: gl.getUniformLocation(positionShader, 'uSpeed'),
            sphere1: gl.getUniformLocation(positionShader, 'uSphere1'),
            sphere2: gl.getUniformLocation(positionShader, 'uSphere2'),
            conf: gl.getUniformLocation(positionShader, 'uConf'),
            twice: gl.getUniformLocation(positionShader, 'uTwice'),
        },
    };

    renderShader = initShaderProgram(renderVS, renderFS);
    renderShaderInfo = {
        program: renderShader,
        attribLocations: {
            texCoor: gl.getAttribLocation(renderShader, 'aTexCoor'),
        },
        uniformLocations: {
            modelViewMatrix: gl.getUniformLocation(renderShader, 'uModelViewMatrix'),
            projectionMatrix: gl.getUniformLocation(renderShader, 'uProjectionMatrix'),
            postitionTexture: gl.getUniformLocation(renderShader, 'uPositionTexture'),
            particle_size: gl.getUniformLocation(renderShader, 'uSize'),
        },
    };

    sphereShader = initShaderProgram(sphereVS, sphereFS);
    sphereShaderInfo = {
        program: sphereShader,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(sphereShader, 'aPosition'),
        },
        uniformLocations: {
            modelViewMatrix: gl.getUniformLocation(sphereShader, 'uModelViewMatrix'),
            projectionMatrix: gl.getUniformLocation(sphereShader, 'uProjectionMatrix'),
            modMat: gl.getUniformLocation(sphereShader, 'uModMat'),
        },
    };
}

function initShaderProgram(vs, fs) {
    const vertexShader = loadShader(gl.VERTEX_SHADER, vs);
    const fragmentShader = loadShader(gl.FRAGMENT_SHADER, fs);

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert('Unable to initialize the shader program: ' + gl.getProgramInfoLog(shaderProgram));
        return null;
    }

    return shaderProgram;
}

function loadShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function initParticles() {
    particles.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, particles.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);

    particles.frameBuffer = gl.createFramebuffer();

    var n = Math.pow(2, Math.ceil(Math.log2(particles.num)));
    var sqrtn = Math.sqrt(n);
    particles.textureWidth = Math.pow(2, Math.ceil(Math.log2(sqrtn)));
    particles.textureHeight = Math.pow(2, Math.floor(Math.log2(sqrtn)));
    console.log("Texture", particles.textureHeight, particles.textureWidth);

    var positions = new Float32Array(particles.textureHeight * particles.textureWidth * 4);
    var texCoor = new Float32Array(particles.textureHeight * particles.textureWidth * 2);

    for (var i = 0; i < particles.num; i++) {
        texCoor[i * 2] = (i % particles.textureWidth) / particles.textureWidth;
        texCoor[i * 2 + 1] = (i / particles.textureWidth) / particles.textureHeight;
        var k = i * 4;
        positions[k] = (Math.random() * 2.0 - 1.0) * 0.1;
        positions[k + 1] = -6.0 * Math.random() - 5.0;
        positions[k + 2] = (Math.random() * 2.0 - 1.0) * 0.1;
        positions[k + 3] = 0.5 + 0.5 * Math.random();
    }
    particles.texCoor = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, particles.texCoor);
    gl.bufferData(gl.ARRAY_BUFFER, texCoor, gl.STATIC_DRAW);

    particles.positionTexture = createTexture(0);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, particles.textureWidth, particles.textureHeight, 0, gl.RGBA, gl.FLOAT, positions);

    particles.positionTextureWrite = createTexture(1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, particles.textureWidth, particles.textureHeight, 0, gl.RGBA, gl.FLOAT, null);

}

function renderParticles() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    var positions = new Float32Array(particles.textureHeight * particles.textureWidth * 4);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, particles.positionTexture);

    for (var i = 0; i < particles.num; i++) {
        var k = i * 4;
        positions[k] = (Math.random() * 2.0 - 1.0) * 0.1;
        positions[k + 1] = -6.0 * Math.random() - 5.0;
        positions[k + 2] = (Math.random() * 2.0 - 1.0) * 0.1;
        positions[k + 3] = 0.5 + 0.5 * Math.random();
    }

    particles.spawnTexture = createTexture(2);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, particles.textureWidth, particles.textureHeight, 0, gl.RGBA, gl.FLOAT, positions);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, particles.spawnTexture);

    // Position shader
    gl.bindBuffer(gl.ARRAY_BUFFER, particles.quadBuffer);
    gl.enableVertexAttribArray(positionShaderInfo.attribLocations.vertexPosition);
    gl.vertexAttribPointer(positionShaderInfo.attribLocations.vertexPosition, 2, gl.FLOAT, false, 0, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, particles.frameBuffer);

    gl.useProgram(positionShaderInfo.program);
    gl.uniform1i(positionShaderInfo.uniformLocations.twice, config.twice);
    gl.uniform1i(positionShaderInfo.uniformLocations.positionTexture, 0);
    gl.uniform1i(positionShaderInfo.uniformLocations.spawnTexture, 2);
    gl.uniform1f(positionShaderInfo.uniformLocations.speed, particles.speed);
    gl.uniform2f(positionShaderInfo.uniformLocations.resolution, particles.textureWidth, particles.textureHeight);
    gl.uniform4f(positionShaderInfo.uniformLocations.sphere1, sphere.pos_x, sphere.pos_y, sphere.pos_z, sphere.radius);
    gl.uniform4f(positionShaderInfo.uniformLocations.sphere2, sphere2.pos_x, sphere2.pos_y, sphere2.pos_z, sphere2.radius);
    gl.uniform4f(positionShaderInfo.uniformLocations.conf,
        particles.ramp_ratio,
        particles.perlin_ratio,
        particles.force_multiply,
        0.0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, particles.positionTextureWrite, 0);
    gl.viewport(0, 0, particles.textureWidth, particles.textureHeight);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.disableVertexAttribArray(positionShaderInfo.attribLocations.vertexPosition);


    // Swap 
    var tmp = particles.positionTextureWrite;
    particles.positionTextureWrite = particles.positionTexture
    particles.positionTexture = tmp;


    // Render shader
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, particles.positionTexture);

    gl.enable(gl.DEPTH_TEST);

    gl.bindBuffer(gl.ARRAY_BUFFER, particles.texCoor);
    gl.enableVertexAttribArray(renderShaderInfo.attribLocations.texCoor);
    gl.vertexAttribPointer(renderShaderInfo.attribLocations.texCoor, 2, gl.FLOAT, false, 0, 0);

    gl.useProgram(renderShaderInfo.program);
    gl.uniform1f(renderShaderInfo.uniformLocations.particle_size, particles.size);
    gl.uniform1i(renderShaderInfo.uniformLocations.positionTexture, 0);
    gl.uniformMatrix4fv(
        renderShaderInfo.uniformLocations.projectionMatrix,
        false,
        camera.projectionMatrix);
    gl.uniformMatrix4fv(
        renderShaderInfo.uniformLocations.modelViewMatrix,
        false,
        camera.modelViewMatrix);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.drawArrays(gl.POINTS, 0, particles.num);

    gl.disableVertexAttribArray(renderShaderInfo.attribLocations.texCoor);
}

function initSphere(sphere) {
    const res_x = 30;
    const res_y = 2 * res_x;

    var positions = [];
    var indices = [];

    sphere.modMat = mat4.create();
    mat4.translate(sphere.modMat, sphere.modMat, [sphere.pos_x, sphere.pos_y, sphere.pos_z]);

    for (var i_x = 0; i_x <= res_x; i_x++) {
        var theta = i_x * Math.PI / res_x;
        var sintheta = Math.sin(theta);
        var costheta = Math.cos(theta);
        for (var i_y = 0; i_y <= res_y; i_y++) {
            var alpha = (i_y * 2 * Math.PI) / res_y;
            var sinalpha = Math.sin(alpha);
            var cosalpha = Math.cos(alpha);
            var x = costheta * cosalpha;
            var y = cosalpha * sintheta;
            var z = sinalpha;
            positions.push(sphere.radius * x);
            positions.push(sphere.radius * y);
            positions.push(sphere.radius * z);
        }
    }

    for (var i_x = 0; i_x < res_x; i_x++) {
        for (var i_y = 0; i_y < res_y; i_y++) {
            var p1 = i_x * (res_y + 1) + i_y;
            var p2 = p1 + res_y + 1;
            indices.push(p1);
            indices.push(p2);
            indices.push(p1 + 1);
            indices.push(p2);
            indices.push(p2 + 1);
            indices.push(p1 + 1);
        }
    }

    sphere.vertexPosition = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, sphere.vertexPosition);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    sphere.vertexIndices = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphere.vertexIndices);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    sphere.numIdx = indices.length;

}

function renderSphere(sphere) {
    gl.useProgram(sphereShaderInfo.program);

    gl.bindBuffer(gl.ARRAY_BUFFER, sphere.vertexPosition);
    gl.enableVertexAttribArray(sphereShaderInfo.attribLocations.vertexPosition);
    gl.vertexAttribPointer(sphereShaderInfo.attribLocations.vertexPosition, 3, gl.FLOAT, false, 0, 0);

    gl.uniformMatrix4fv(
        sphereShaderInfo.uniformLocations.projectionMatrix,
        false,
        camera.projectionMatrix);
    gl.uniformMatrix4fv(
        sphereShaderInfo.uniformLocations.modelViewMatrix,
        false,
        camera.modelViewMatrix);
    gl.uniformMatrix4fv(sphereShaderInfo.uniformLocations.modMat,
        false,
        sphere.modMat);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphere.vertexIndices);
    gl.drawElements(gl.TRIANGLES, sphere.numIdx, gl.UNSIGNED_SHORT, 0);
}

function initCamera() {
    const fieldOfView = 45 * Math.PI / 180;   // in radians
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const zNear = 0.1;
    const zFar = 100.0;
    const projectionMatrix = mat4.create();

    mat4.perspective(projectionMatrix,
        fieldOfView,
        aspect,
        zNear,
        zFar);

    const modelViewMatrix = mat4.create();
    mat4.translate(modelViewMatrix,
        modelViewMatrix,
        [-0.0, 0.0, -20.0]);

    camera = {
        projectionMatrix,
        modelViewMatrix,
    }
}

function createTexture(index) {
    var texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    return texture;
}

var mouseDown = false;
var lastMouseX = 0;
var lastMouseY = 0;

var CAMERA_SENSITIVITY = 0.004;

canvas.addEventListener('mousedown', (event) => {
    mouseDown = true;
    lastMouseX = event.x;
    lastMouseY = event.y;
});

document.addEventListener('mouseup', (event) => {
    mouseDown = false;
});

canvas.addEventListener('mousemove', (event) => {
    if (mouseDown) {
        var x = event.x;
        var y = event.y;

        var dx = (x - lastMouseX) * CAMERA_SENSITIVITY;
        var dy = (y - lastMouseY) * CAMERA_SENSITIVITY;

        mat4.rotateX(camera.modelViewMatrix, camera.modelViewMatrix, dx);
        mat4.rotateY(camera.modelViewMatrix, camera.modelViewMatrix, dy);

        lastMouseX = x;
        lastMouseY = y;
    }
});

canvas.addEventListener('mousewheel', function (event) {
    console.log(0)
    var delta = event.wheelDelta / 120
    if (delta < 0)
        mat4.translate(camera.modelViewMatrix, camera.modelViewMatrix, [0.0, 0.0, -0.25]);
    else
        mat4.translate(camera.modelViewMatrix, camera.modelViewMatrix, [0.0, 0.0, +0.25]);
})



import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FaceLandmarker, FilesetResolver } from 'https://esm.sh/@mediapipe/tasks-vision@0.10.3';

const statusDiv = document.getElementById('status');
const container = document.getElementById('container');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202020); // Gris oscuro

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 0.5);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
dirLight.position.set(0, 1, 2);
scene.add(dirLight);

// 2. VARIABLES
let headModel = null;
let headMeshes = []; // Array para guardar todas las mallas (cabeza, dientes, ojos...)
let faceLandmarker;
let video;
let lastVideoTime = -1;

// 3. CARGAR MODELO
const loader = new GLTFLoader();
loader.load(
    'papa_model_head.glb',
    (gltf) => {
        headModel = gltf.scene;
        scene.add(headModel);

        // Ajuste de posición y escala automático para asegurar que se vea
        const box = new THREE.Box3().setFromObject(headModel);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        // Escalar para que mida 1 unidad de alto aprox
        const maxDim = Math.max(size.x, size.y, size.z);
        // USUARIO: "escala del 90%" -> 0.5 * 0.9 = 0.45
        const scale = 0.45 / maxDim;
        headModel.scale.set(scale, scale, scale);

        // Centrar en el origen
        box.setFromObject(headModel);
        box.getCenter(center);
        headModel.position.sub(center);

        // Buscar Blendshapes en TODAS las mallas
        headModel.traverse((child) => {
            if (child.isMesh && child.morphTargetInfluences) {
                headMeshes.push(child);
                console.log("Malla con blendshapes añadida:", child.name);
            }
        });

        statusDiv.textContent = "Modelo cargado y escalado (90%). Iniciando cámara...";

        // Start animation loop immediately
        animate();

        // Intentar iniciar MediaPipe
        initializeFaceTracking();
    },
    undefined,
    (error) => {
        console.error(error);
        statusDiv.textContent = "Error 404: No encuentro el archivo .glb";
    }
);

// 4. INICIALIZAR MEDIAPIPE
async function initializeFaceTracking() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "CPU" // CAMBIO: CPU es más estable en móviles que GPU
            },
            runningMode: "VIDEO",
            outputFaceGeometry: true, // Esto habilita facialTransformationMatrixes
            outputFaceBlendshapes: true,
            numFaces: 1
        });

        video = document.createElement('video');
        video.setAttribute('autoplay', '');
        video.setAttribute('playsinline', '');

        // OPTIMIZACIÓN: Bajar resolución para que vaya más rápido
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30, max: 30 }
            }
        });
        video.srcObject = stream;

        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve();
            };
        });

        statusDiv.textContent = "SISTEMA ONLINE. Mueve tu cara.";

    } catch (error) {
        console.error(error);
        statusDiv.textContent = "Error MediaPipe: " + error.message;
    }
}

// 6. LOOP
function animate() {
    requestAnimationFrame(animate);

    try {
        if (faceLandmarker && video && video.readyState >= 2) {
            if (video.currentTime !== lastVideoTime) {
                lastVideoTime = video.currentTime;
                let results = faceLandmarker.detectForVideo(video, lastVideoTime);

                if (results && headModel) {
                    // A) ROTACIÓN
                    if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
                        const matrix = new THREE.Matrix4().fromArray(results.facialTransformationMatrixes[0].data);
                        const rotation = new THREE.Euler().setFromRotationMatrix(matrix);
                        headModel.rotation.set(rotation.x, -rotation.y, -rotation.z);
                    }

                    // B) EXPRESIONES
                    if (headMeshes.length > 0 && results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                        const blendshapes = results.faceBlendshapes[0].categories;

                        // Iterar sobre todas las mallas
                        headMeshes.forEach(mesh => {
                            const influenceMap = mesh.morphTargetDictionary;

                            // Mapeo directo: MediaPipe Name -> GLB Morph Target Name
                            blendshapes.forEach(shape => {
                                const shapeName = shape.categoryName;
                                if (influenceMap[shapeName] !== undefined) {
                                    mesh.morphTargetInfluences[influenceMap[shapeName]] = shape.score;
                                }
                            });
                        });
                    }
                }
            }
        }
    } catch (e) {
        console.error("Error en loop:", e);
    }
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
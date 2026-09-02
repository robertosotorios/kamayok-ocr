https://www.runpod.io/pricing
https://docling.ai/



curl -X POST "https://api.runpod.ai/v2/TU_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TU_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "pdf_url": "https://pub-xxxx.r2.dev/factura-ejemplo.pdf"
    }
  }'
  
  
  
''' 
  const RUNPOD_ENDPOINT_ID = "tu-endpoint-id";
const RUNPOD_API_KEY = env.RUNPOD_API_KEY;

// URL pública o prefirmada del archivo en Cloudflare R2
const fileUrl = "https://pub-xxxx.r2.dev/facturas/factura-001.pdf";

const response = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${RUNPOD_API_KEY}`
  },
  body: JSON.stringify({
    input: {
      pdf_url: fileUrl
    }
  })
});

const data = await response.json();

if (data.status === "COMPLETED") {
  console.log("Markdown extraído:", data.output.markdown);
} else {
  console.error("Error en el job:", data);
}

'''




'''
import * as fs from "node:fs/promises";

interface RunPodSyncResponse {
  id: string;
  status: "COMPLETED" | "FAILED" | "IN_PROGRESS";
  output?: {
    markdown?: string;
    status?: string;
    error?: string;
  };
  error?: string;
}

const RUNPOD_ENDPOINT_ID = "tu-endpoint-id";
const RUNPOD_API_KEY = "tu_api_key_aqui";
const FILE_PATH = "./factura.pdf";

async function processPdfBase64(): Promise<void> {
  // 1. Leer el archivo binario y convertirlo a Base64
  const fileBuffer = await fs.readFile(FILE_PATH);
  const pdfBase64 = fileBuffer.toString("base64");

  // 2. Enviar petición síncrona a RunPod
  const response = await fetch(`https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${RUNPOD_API_KEY}`
    },
    body: JSON.stringify({
      input: {
        pdf_base64: pdfBase64
      }
    })
  });

  const data = (await response.json()) as RunPodSyncResponse;

  if (data.status === "COMPLETED" && data.output?.markdown) {
    console.log("Extracción completada con éxito:\n");
    console.log(data.output.markdown);
  } else {
    console.error("Error al procesar el documento:", data.error ?? data.output?.error);
  }
}

processPdfBase64().catch(console.error);
'''


'''
export interface Env {
  RUNPOD_ENDPOINT_ID: string;
  RUNPOD_API_KEY: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method !== "POST") {
      return new Response("Envía un POST con el archivo", { status: 405 });
    }

    // 1. Leer el archivo binario del cuerpo de la petición
    const arrayBuffer = await request.arrayBuffer();

    // 2. Convertir ArrayBuffer a Base64 (estándar Web API en Workers)
    const uint8Array = new Uint8Array(arrayBuffer);
    let binaryString = "";
    for (let i = 0; i < uint8Array.byteLength; i++) {
      binaryString += String.fromCharCode(uint8Array[i]);
    }
    const pdfBase64 = btoa(binaryString);

    // 3. Llamar al endpoint de RunPod
    const response = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/runsync`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${env.RUNPOD_API_KEY}`
        },
        body: JSON.stringify({
          input: {
            pdf_base64: pdfBase64
          }
        })
      }
    );

    const result = await response.json();
    return Response.json(result);
  }
};
'''

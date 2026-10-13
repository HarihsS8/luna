import JSZip from 'jszip';
import { LUNA_PYTHON_FILES } from './pythonFiles';

/**
 * Packages all LUNA Python files into a single downloadable ZIP file.
 */
export async function downloadLunaPythonZip(): Promise<void> {
  const zip = new JSZip();

  // Add each file to the zip archive preserving folder structure
  for (const file of LUNA_PYTHON_FILES) {
    zip.file(file.path, file.code);
  }

  // Generate the zip blob
  const content = await zip.generateAsync({ type: 'blob' });

  // Trigger download in browser
  const url = URL.createObjectURL(content);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'luna-bert-transformer-python.zip';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

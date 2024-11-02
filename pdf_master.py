import os
import fitz
import shutil
import time
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint
import tempfile
import numpy as np
from PIL import Image
import io
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

class PDFMaster:
    def __init__(self):
        self.console = Console()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_dir = os.path.join(self.base_dir, "input")
        self.output_dir = os.path.join(self.base_dir, "output")
        self.temp_dir = os.path.join(tempfile.gettempdir(), "pdf_master_temp")
        self.setup_folders()

    def setup_folders(self):
        """ফোল্ডার স্ট্রাকচার সেটআপ"""
        for folder in [self.input_dir, self.output_dir, self.temp_dir]:
            os.makedirs(folder, exist_ok=True)

    def cleanup(self):
        """টেম্পোরারি ফাইল ক্লিনআপ"""
        try:
            shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir)
        except Exception as e:
            self.console.print(f"[red]Cleanup error: {str(e)}")

    def get_pdf_list(self):
        """PDF ফাইল লিস্ট"""
        files = []
        for file in os.listdir(self.input_dir):
            if file.lower().endswith('.pdf'):
                path = os.path.join(self.input_dir, file)
                try:
                    doc = fitz.open(path)
                    files.append({
                        'name': file,
                        'path': path,
                        'pages': doc.page_count,
                        'size': os.path.getsize(path) / (1024 * 1024)  # MB
                    })
                    doc.close()
                except Exception:
                    continue
        return files

    def show_pdf_list(self):
        """PDF লিস্ট দেখানো"""
        files = self.get_pdf_list()
        if not files:
            self.console.print("[red]No PDF files found in input folder!")
            return None

        table = Table(title="Available PDF Files")
        table.add_column("No.", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Pages", justify="right")
        table.add_column("Size (MB)", justify="right")

        for i, file in enumerate(files, 1):
            table.add_row(
                str(i),
                file['name'],
                str(file['pages']),
                f"{file['size']:.2f}"
            )

        self.console.print(table)
        return files

    def merge_pdfs(self, files, indices, output_name):
        """PDF মার্জ"""
        try:
            merged_pdf = fitz.open()
            total_pages = sum(files[i]['pages'] for i in indices)

            with Progress() as progress:
                task = progress.add_task("[cyan]Merging PDFs...", total=total_pages)
                
                for idx in indices:
                    file = files[idx]
                    doc = fitz.open(file['path'])
                    merged_pdf.insert_pdf(doc)
                    progress.update(task, advance=doc.page_count)
                    doc.close()

            output_path = os.path.join(self.output_dir, output_name)
            merged_pdf.save(output_path, garbage=4, deflate=True)
            merged_pdf.close()
            
            self.console.print(f"[green]Successfully merged to: {output_name}")
            return output_path
            
        except Exception as e:
            self.console.print(f"[red]Error merging PDFs: {str(e)}")
            return None

    def invert_pdf(self, input_path, output_path):
        """PDF ইনভার্ট - অপটিমাইজড ভার্সন"""
        try:
            doc = fitz.open(input_path)
            out_pdf = fitz.open()
            
            # কপি মেটাডেটা
            out_pdf.metadata = doc.metadata
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Inverting pages...", total=doc.page_count)
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Get pixmap with optimized resolution
                    zoom = 1.0  # Lower resolution for better performance
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    # Convert to numpy array for faster processing
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                    img_array = img_array.reshape(pix.height, pix.width, 3)
                    
                    # Quick check for dark background
                    is_dark = np.mean(img_array) < 128
                    
                    if is_dark:
                        # Fast inversion
                        img_array = 255 - img_array
                        
                        # Convert to bytes
                        img = Image.fromarray(img_array)
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='JPEG', 
                               quality=85,  # Lower quality for smaller size
                               optimize=True)
                        
                        # Create new page
                        new_page = out_pdf.new_page(width=page.rect.width,
                                                  height=page.rect.height)
                        
                        # Insert inverted image
                        new_page.insert_image(page.rect, 
                                            stream=img_bytes.getvalue(),
                                            keep_proportion=True)
                    else:
                        # Copy original page
                        out_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    progress.update(task, advance=1)
            
            # Save with optimization
            out_pdf.save(output_path,
                        garbage=4,
                        deflate=True,
                        clean=True,
                        linear=True)
            
            # Print stats
            orig_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            self.console.print(f"\n[green]Successfully inverted: {os.path.basename(output_path)}")
            self.console.print(f"Original size: {orig_size:.1f} MB")
            self.console.print(f"New size: {new_size:.1f} MB")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error inverting PDF: {str(e)}")
            return False
            
        finally:
            if 'doc' in locals(): doc.close()
            if 'out_pdf' in locals(): out_pdf.close()

    def invert_udvash_pdf(self, input_path, output_path):
        """PDF ইনভার্ট - উদ্ভাস স্লাইড অপটিমাইজড"""
        try:
            doc = fitz.open(input_path)
            out_pdf = fitz.open()
            
            # কপি মেটাডেটা
            out_pdf.metadata = doc.metadata
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Inverting Udvash slides...", total=doc.page_count)
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    zoom = 1.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                    img_array = img_array.reshape(pix.height, pix.width, 3)
                    
                    # Convert to HSV for better color detection
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    
                    # Create mask for dark blue background (#090d16)
                    lower_blue = np.array([100, 50, 0])  # Dark blue in HSV
                    upper_blue = np.array([130, 255, 30])
                    mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    
                    # Create inverted image
                    inverted = 255 - img_array
                    
                    # Modify colors specifically for areas with dark blue background
                    inverted[mask > 0] = [255, 253, 245]  # Slightly warm white
                    
                    # Convert to bytes
                    img = Image.fromarray(inverted)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='JPEG', quality=85, optimize=True)
                    
                    # Create new page
                    new_page = out_pdf.new_page(width=page.rect.width,
                                              height=page.rect.height)
                    
                    # Insert modified image
                    new_page.insert_image(page.rect, 
                                        stream=img_bytes.getvalue(),
                                        keep_proportion=True)
                    
                    progress.update(task, advance=1)
                
            # Save with optimization
            out_pdf.save(output_path,
                        garbage=4,
                        deflate=True,
                        clean=True,
                        linear=True)
            
            # Print stats
            orig_size = os.path.getsize(input_path) / (1024 * 1024)
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            
            self.console.print(f"\n[green]Successfully inverted Udvash slides: {os.path.basename(output_path)}")
            self.console.print(f"Original size: {orig_size:.1f} MB")
            self.console.print(f"New size: {new_size:.1f} MB")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error inverting PDF: {str(e)}")
            return False
            
        finally:
            if 'doc' in locals(): doc.close()
            if 'out_pdf' in locals(): out_pdf.close()

    def analyze_page_parallel(self, args):
        """Single page analysis for parallel processing"""
        try:
            page, zoom = args
            # Create a new fitz Matrix for this thread
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.height, pix.width, 3)
            
            # Basic analysis without OpenCV
            # Convert to grayscale using numpy
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            
            # Simple thresholding
            binary = gray < 250
            
            # Calculate content percentage
            content_percentage = np.mean(binary)
            
            # Simplified analysis
            is_empty = content_percentage < 0.03
            
            return {
                'center_content': content_percentage,
                'border_content': 0,  # Simplified
                'num_components': 0,  # Simplified
                'is_empty': is_empty
            }
            
        except Exception as e:
            print(f"Error analyzing page: {str(e)}")
            return None
        finally:
            # Clean up
            if 'pix' in locals():
                pix = None

    def remove_empty_pages(self, input_path, output_path):
        """Optimized empty page removal"""
        try:
            doc = fitz.open(input_path)
            new_doc = fitz.open()
            removed_doc = fitz.open()
            empty_pages = []
            
            # Prepare pages
            pages = [(doc[i], 2.0) for i in range(doc.page_count)]
            
            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Analyzing pages...", 
                    total=doc.page_count
                )
                
                # Process pages sequentially for stability
                results = []
                for page_data in pages:
                    result = self.analyze_page_parallel(page_data)
                    results.append(result)
                    progress.update(task, advance=1)
                
                # Process results and build PDFs
                for page_num, analysis in enumerate(results):
                    if analysis and not analysis['is_empty']:
                        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    else:
                        empty_pages.append(page_num + 1)
                        removed_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        
                        if analysis:
                            self.console.print(f"\nPage {page_num + 1} appears empty:")
                            self.console.print(f"- Content: {analysis['center_content']:.4f}")
            
            if empty_pages:
                # Backup original
                backup_dir = os.path.join(self.output_dir, "backup")
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(input_path))
                shutil.copy2(input_path, backup_path)
                
                # Save processed PDF
                new_doc.save(
                    output_path,
                    garbage=4,
                    deflate=True,
                    clean=True,
                    linear=True
                )
                
                # Save removed pages
                removed_pages_path = os.path.join(
                    self.output_dir,
                    f"{os.path.splitext(os.path.basename(input_path))[0]}_removed_pages.pdf"
                )
                removed_doc.save(
                    removed_pages_path,
                    garbage=4,
                    deflate=True,
                    clean=True
                )
                
                self.console.print(f"\n[green]Removed {len(empty_pages)} empty pages")
                self.console.print(f"Empty pages: {empty_pages}")
                self.console.print(f"Removed pages saved to: {removed_pages_path}")
            else:
                shutil.copy2(input_path, output_path)
                self.console.print("\n[yellow]No empty pages found")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error removing empty pages: {str(e)}")
            return False
            
        finally:
            if 'doc' in locals(): doc.close()
            if 'new_doc' in locals(): new_doc.close()
            if 'removed_doc' in locals(): removed_doc.close()

    def process_file(self, file_path, operations):
        """একটি ফাইল প্রসেস"""
        current_file = file_path
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        try:
            for op in operations:
                temp_output = os.path.join(self.temp_dir, f"{base_name}_{op}.pdf")
                
                if op == "invert":
                    if not self.invert_pdf(current_file, temp_output):
                        return None
                elif op == "remove_empty":
                    if not self.remove_empty_pages(current_file, temp_output):
                        return None
                
                current_file = temp_output
            
            # Move final result to output
            output_path = os.path.join(self.output_dir, f"{base_name}_processed.pdf")
            shutil.move(current_file, output_path)
            return output_path
            
        except Exception as e:
            self.console.print(f"[red]Error processing {file_name}: {str(e)}")
            return None

    def run(self):
        """মেইন প্রোগ্রাম"""
        while True:
            self.console.clear()
            self.console.print("[bold cyan]PDF Master Tool v3.1", justify="center")
            self.console.print("=" * 50, justify="center")
            
            # Show main menu
            menu = Table(show_header=True, header_style="bold magenta")
            menu.add_column("Option", style="cyan", width=12)
            menu.add_column("Description", style="green")
            
            menu.add_row("1", "Single Operation Mode")
            menu.add_row("2", "Automation Mode")
            menu.add_row("3", "Exit")
            
            self.console.print(menu)
            choice = Prompt.ask("Select mode", choices=["1", "2", "3"])
            
            if choice == "3":
                break
            
            files = self.show_pdf_list()
            if not files:
                break
            
            if choice == "1":
                # Single operation mode
                operations = []
                
                # Show operation menu
                op_menu = Table(show_header=True)
                op_menu.add_column("Operation", style="cyan")
                op_menu.add_column("Description", style="green")
                
                op_menu.add_row("1", "Merge PDFs")
                op_menu.add_row("2", "Invert Dark Pages")
                op_menu.add_row("3", "Remove Empty Pages")
                
                self.console.print(op_menu)
                op_choice = Prompt.ask("Select operation", choices=["1", "2", "3"])
                
                if op_choice == "1":
                    # Merge PDFs
                    indices = []
                    while True:
                        idx = Prompt.ask("Enter file number (or 'done')")
                        if idx.lower() == 'done':
                            break
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(files):
                                indices.append(idx)
                            else:
                                self.console.print("[red]Invalid file number!")
                        except ValueError:
                            self.console.print("[red]Invalid input!")
                    
                    if indices:
                        output_name = Prompt.ask("Enter output filename", default="merged.pdf")
                        if not output_name.lower().endswith('.pdf'):
                            output_name += '.pdf'
                        self.merge_pdfs(files, indices, output_name)
                
                elif op_choice == "2":
                    # Show invert options
                    invert_menu = Table(show_header=True)
                    invert_menu.add_column("Option", style="cyan")
                    invert_menu.add_column("Description", style="green")
                    
                    invert_menu.add_row("1", "Normal Invert")
                    invert_menu.add_row("2", "Udvash Slides Invert")
                    
                    self.console.print(invert_menu)
                    invert_choice = Prompt.ask("Select invert type", choices=["1", "2"])
                    
                    # Get file selection
                    indices = []
                    while True:
                        idx = Prompt.ask("Enter file number (or 'done')")
                        if idx.lower() == 'done':
                            break
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(files):
                                indices.append(idx)
                            else:
                                self.console.print("[red]Invalid file number!")
                        except ValueError:
                            self.console.print("[red]Invalid input!")
                    
                    if indices:
                        for idx in indices:
                            file = files[idx]
                            output_path = os.path.join(
                                self.output_dir, 
                                f"{os.path.splitext(file['name'])[0]}_inverted.pdf"
                            )
                            
                            if invert_choice == "1":
                                self.invert_pdf(file['path'], output_path)
                            else:
                                self.invert_udvash_pdf(file['path'], output_path)
                
                elif op_choice == "3":
                    # Remove empty pages from selected files
                    indices = []
                    while True:
                        idx = Prompt.ask("Enter file number (or 'done')")
                        if idx.lower() == 'done':
                            break
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(files):
                                indices.append(idx)
                            else:
                                self.console.print("[red]Invalid file number!")
                        except ValueError:
                            self.console.print("[red]Invalid input!")
                    
                    if indices:
                        for idx in indices:
                            file = files[idx]
                            output_path = os.path.join(self.output_dir, f"{os.path.splitext(file['name'])[0]}_cleaned.pdf")
                            self.remove_empty_pages(file['path'], output_path)
            
            else:
                # Automation mode
                self.console.print("\n[cyan]Automation Mode")
                self.console.print("This will process selected files in sequence: Merge (optional) -> Invert -> Remove Empty")
                
                # First ask for merge
                if Confirm.ask("Do you want to merge PDFs first?"):
                    indices = []
                    while True:
                        idx = Prompt.ask("Enter file number (or 'done')")
                        if idx.lower() == 'done':
                            break
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(files):
                                indices.append(idx)
                            else:
                                self.console.print("[red]Invalid file number!")
                        except ValueError:
                            self.console.print("[red]Invalid input!")
                    
                    if indices:
                        # Merge files
                        output_name = Prompt.ask("Enter output filename", default="merged.pdf")
                        if not output_name.lower().endswith('.pdf'):
                            output_name += '.pdf'
                        
                        merged_path = self.merge_pdfs(files, indices, output_name)
                        if merged_path:
                            # Process merged file
                            temp_path = os.path.join(self.temp_dir, "temp_inverted.pdf")
                            if self.invert_pdf(merged_path, temp_path):
                                output_path = os.path.join(self.output_dir, f"{os.path.splitext(output_name)[0]}_processed.pdf")
                                self.remove_empty_pages(temp_path, output_path)
                else:
                    # Process individual files
                    indices = []
                    while True:
                        idx = Prompt.ask("Enter file number (or 'done')")
                        if idx.lower() == 'done':
                            break
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(files):
                                indices.append(idx)
                            else:
                                self.console.print("[red]Invalid file number!")
                        except ValueError:
                            self.console.print("[red]Invalid input!")
                    
                    if indices:
                        for idx in indices:
                            file = files[idx]
                            # First invert
                            temp_path = os.path.join(self.temp_dir, f"{os.path.splitext(file['name'])[0]}_temp.pdf")
                            if self.invert_pdf(file['path'], temp_path):
                                # Then remove empty pages
                                output_path = os.path.join(self.output_dir, f"{os.path.splitext(file['name'])[0]}_processed.pdf")
                                self.remove_empty_pages(temp_path, output_path)
            
            if not Confirm.ask("Process more files?"):
                break
        
        self.cleanup()
        self.console.print("[green]All operations completed!")

if __name__ == "__main__":
    PDFMaster().run() 
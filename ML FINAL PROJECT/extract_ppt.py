import zipfile
import re

def extract_text_from_pptx(pptx_path):
    try:
        with zipfile.ZipFile(pptx_path, 'r') as z:
            # Also get notes if any
            notes_files = [f for f in z.namelist() if f.startswith('ppt/notesSlides/notesSlide') and f.endswith('.xml')]
            
            slide_files = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            slide_files.sort(key=lambda x: int(x.replace('ppt/slides/slide', '').replace('.xml', '')))
            
            for slide_file in slide_files:
                xml_content = z.read(slide_file).decode('utf-8')
                texts = re.findall(r'<a:t[^>]*>(.*?)</a:t>', xml_content)
                print(f"=== {slide_file} ===")
                # Clean up xml entities
                clean_texts = [t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') for t in texts]
                print(" | ".join(clean_texts))
                
                # Check for corresponding notes slide
                slide_num = slide_file.replace('ppt/slides/slide', '').replace('.xml', '')
                notes_file = f'ppt/notesSlides/notesSlide{slide_num}.xml'
                if notes_file in notes_files:
                    notes_xml = z.read(notes_file).decode('utf-8')
                    notes_texts = re.findall(r'<a:t[^>]*>(.*?)</a:t>', notes_xml)
                    clean_notes = [t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') for t in notes_texts]
                    print("--- NOTES ---")
                    print(" | ".join(clean_notes))
                
                print()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    extract_text_from_pptx('Noon_Strategic_Intelligence.pptx')

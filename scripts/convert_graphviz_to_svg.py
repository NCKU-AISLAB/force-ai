import os
import re
from graphviz import Source

def extract_graphviz_blocks(md_content):
    pattern = r'```dot\n(.*?)\n\s*```'
    return list(re.finditer(pattern, md_content, re.DOTALL))

def convert_to_svg(dot_code, svg_path):
    try:
        graph = Source(dot_code)
        graph.format = 'svg'
        graph.render(svg_path, cleanup=True)
    except Exception as e:
        print(f"Error converting to SVG: {e}")

def process_markdown_file(md_path, output_dir, output_markdown):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = extract_graphviz_blocks(content)
    new_content = content
    offset = 0  # To adjust positions after inserting SVG tags

    for i, match in enumerate(matches):
        dot_code = match.group(1)
        svg_filename = f"{os.path.splitext(os.path.basename(md_path))[0]}_graphviz_{i}.svg"
        svg_path = os.path.join(output_dir, svg_filename)

        convert_to_svg(dot_code, os.path.splitext(svg_path)[0])  # graphviz auto adds .svg

        # Build replacement text
        comment = f"<!--{dot_code}-->\n"
        img_tag = f"\n<img src=\"{output_dir}/{svg_filename}\" alt=\"graphviz_{i}\">\n"

        start, end = match.start() + offset, match.end() + offset
        new_content = new_content[:start] + comment + img_tag + new_content[end:]

        offset += len(comment + img_tag) - (end - start)

    # Save updated markdown
    with open(output_markdown, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Processed {md_path} → {output_markdown}")

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("md_file", help="Path to the Markdown file")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory for SVGs")
    parser.add_argument("-m", "--output-markdown", default="default.md", help="Output filename for markdown")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    process_markdown_file(args.md_file, args.output_dir, args.output_markdown)

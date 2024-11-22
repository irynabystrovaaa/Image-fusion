import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2 
import gradio as gr
from io import BytesIO
from PIL import Image
import colorsys
import os

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class GradientalCorrection:
    def __init__(self, image):
        self.image = image.ravel()
        self.counter = Counter(self.image)

    def prob(self, a):
        # returns the probability of a given number a
        return float(self.counter[a]) / len(self.image)

    @staticmethod
    def remove_duplicates(lst):
        return [t for t in (set(tuple(i) for i in lst))]

    def val_prob_min(self, perc):
        ans = [(v, self.prob(v)) for v in self.image]
        cleans_ans = self.remove_duplicates(ans)
        sorted_ans = sorted(cleans_ans, key=lambda x: x[0])
        A = [i for i in sorted_ans if i[1] >= perc]
        return A[0]

    def val_prob_max(self, perc):
        ans = [(v, self.prob(v)) for v in self.image]
        cleans_ans = self.remove_duplicates(ans)
        sorted_ans = sorted(cleans_ans, key=lambda x: x[0], reverse=True)
        A = [i for i in sorted_ans if i[1] <= perc]
        
        if len(A) < 1:
            return sorted_ans[0]
        return A

    def gradiental_correction(self, t=0.0003):
        new_img = []
        min_val = self.val_prob_min(t)[0]
        max_val = self.val_prob_max(t)[0][0]
        
        for i in self.image:
            if i < min_val:
                new_img.append(0)
            elif min_val < i < max_val:
                val = round((255 * (i - min_val)) / (max_val - min_val))
                new_img.append(val)
            else:
                new_img.append(255)
                
        return np.array(new_img).reshape(480, 720)

# Допоміжні функції

def load_image_pil(image):
    """
    Завантажуємо зображення у форматі grayscale.
    """
    if image is None:
        raise ValueError("No image uploaded.")
    image = image.convert('L')  # Перетворюємо в grayscale
    return np.array(image)

def show_image_with_histogram(image, title='Зображення', title_left = 'Гістограма'):
    """
    Створює зображення з його гістограмою.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(title)
    axs[0].axis('off')
    
    axs[1].hist(image.ravel(), bins=256, range=(0, 255), color='black')
    axs[1].set_title(title_left)
    axs[1].set_xlim([0, 255])
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def scale_image(image):
    """
    Масштабування зображення до діапазону 0-255.
    """
    image_min = image.min()
    image_max = image.max()
    if image_max == image_min:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = 255 * (image - image_min) / (image_max - image_min)
    return scaled.astype(np.uint8)

def fusion_methods(Vc, IRc):
    """
    Виконує комплексування зображень за різними алгоритмами.
    Повертає словник з результатами, включаючи Ft1 та Ft2.
    """
    # Avoid division by zero by adding a small epsilon where necessary
    epsilon = 1e-10

    F = {
        'Fmax': np.maximum(Vc, IRc),
        'Fmin': np.minimum(Vc, IRc),
        'Fc': np.cbrt((Vc**3 + IRc**3) / 2),
        'Fq': np.sqrt((Vc**2 + IRc**2) / 2),
        'Fa': (Vc + IRc) / 2,
        'Fg': np.sqrt(Vc * IRc),
        'Fh': (2 * Vc * IRc) / (Vc + IRc + epsilon),  # Додаємо малу константу щоб уникнути ділення на нуль
        'Fch': (Vc**2 + IRc**2) / (Vc + IRc + epsilon),
        # Нові методи комплексування
        'Ft1': (np.arctan(IRc / (Vc + epsilon)) * 255 * 2 / np.pi).astype(np.float64),
        'Ft2': (np.arctan(Vc / (IRc + epsilon)) * 255 * 2 / np.pi).astype(np.float64),
    }
    
    # Масштабування кожного комплексованого зображення
    for key in F:
        F[key] = scale_image(F[key])
    
    return F

def create_colormap(palette):
    """
    Створює кастомну колірну палітру на основі заданої назви.
    Реалізація кольорових палітр згідно з магістерською роботою.
    """
    if palette == 'Grayscale':
        return mcolors.LinearSegmentedColormap.from_list('Grayscale', ['black', 'white'], N=256)
    
    elif palette == 'Rainbow':
        # Розбиття палітри HSL на 7 кольорів від червоного до фіолетового
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'Rainbow',
            ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
            N=256
        )
        return cmap
    
    elif palette == 'Fire':
        # Від червоного до жовтого з HSL (0 to 0.23)
        colors = []
        for i in np.linspace(0, 0.23, 256):
            r, g, b = colorsys.hls_to_rgb(i, 0.5, 1)
            colors.append((r, g, b))
        return mcolors.LinearSegmentedColormap.from_list('Fire', colors, N=256)
    
    elif palette == 'Thermometer':
        # Синій, блакитний, зелений, жовтий, червоний
        colors = [
            colorsys.hls_to_rgb(0.68, 0.5, 1),  # Blue
            colorsys.hls_to_rgb(0.49, 0.5, 1),  # Cyan
            colorsys.hls_to_rgb(0.35, 0.5, 1),  # Green
            colorsys.hls_to_rgb(0.16, 0.5, 1),  # Yellow
            colorsys.hls_to_rgb(0.01, 0.5, 1)   # Red
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list('Thermometer', colors, N=256)
        return cmap
    
    elif palette == 'Watermelon':
        # Червоний до зеленого
        colors = []
        for i in np.linspace(0, 0.27, 128):
            r, g, b = colorsys.hls_to_rgb(0.0, 0.5, 1)  # Red
            colors.append((r, g, b))
        for i in np.linspace(0.27, 0.35, 128):
            r, g, b = colorsys.hls_to_rgb(0.33, 0.5, 1)  # Green
            colors.append((r, g, b))
        return mcolors.LinearSegmentedColormap.from_list('Watermelon', colors, N=256)
    
    elif palette == 'Temperature':
        # Помаранчевий та блакитний
        colors = []
        for i in np.linspace(0.0, 0.08, 128):
            r, g, b = colorsys.hls_to_rgb(0.075, 0.5, 1)  # Orange
            colors.append((r, g, b))
        for i in np.linspace(0.08, 0.16, 128):
            r, g, b = colorsys.hls_to_rgb(0.48, 0.5, 1)  # Blue
            colors.append((r, g, b))
        return mcolors.LinearSegmentedColormap.from_list('Temperature', colors, N=256)
    
    elif palette == 'Weather':
        # Фіолетовий, синій, блакитний, зелений, світло-зелений, жовтий, темно-жовтий, помаранчевий, червоний
        colors = [
            colorsys.hls_to_rgb(0.77, 0.5, 1),  # Violet
            colorsys.hls_to_rgb(0.68, 0.5, 1),  # Blue
            colorsys.hls_to_rgb(0.51, 0.5, 1),  # Cyan
            colorsys.hls_to_rgb(0.30, 0.5, 1),  # Green
            colorsys.hls_to_rgb(0.35, 0.5, 1),  # Light Green
            colorsys.hls_to_rgb(0.16, 0.5, 1),  # Yellow
            colorsys.hls_to_rgb(0.14, 0.5, 1),  # Dark Yellow
            colorsys.hls_to_rgb(0.07, 0.5, 1),  # Orange
            colorsys.hls_to_rgb(0.01, 0.5, 1)   # Red
        ]
        return mcolors.ListedColormap(colors, name='Weather')
    
    elif palette == 'Pumpkin':
        # Жовтий, темно-помаранчевий, помаранчевий
        colors = [
            colorsys.hls_to_rgb(0.12, 0.5, 1),  # Yellow
            colorsys.hls_to_rgb(0.08, 0.5, 1),  # Dark Orange
            colorsys.hls_to_rgb(0.075, 0.5, 1)  # Orange
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list('Pumpkin', colors, N=256)
        return cmap
    
    else:
        return mcolors.LinearSegmentedColormap.from_list('Default', ['black', 'white'], N=256)

def apply_color_palette(image, palette='Grayscale'):
    """
    акладає кольорову палітру на монохромне зображення.
    Палітри: Grayscale, Rainbow, Fire, Thermometer, Watermelon, Temperature, Weather, Pumpkin
    """
    colormap = create_colormap(palette)
    colored_image = colormap(image / 255.0)[:, :, :3]  # Ігноруємо альфа-канал
    colored_image = (colored_image * 255).astype(np.uint8)
    return colored_image

def synthesize_rgb(IRc, F, scheme):
    """
    Синтезує RGB зображення за заданою схемою.
    scheme - літера від 'a' до 'f'
    """
    if scheme == 'a':
        RGB = np.stack([IRc, IRc, F], axis=2)
    elif scheme == 'b':
        RGB = np.stack([IRc, F, F], axis=2)
    elif scheme == 'c':
        RGB = np.stack([IRc, IRc, F // 2], axis=2)
    elif scheme == 'd':
        RGB = np.stack([IRc, F, F // 2], axis=2)
    elif scheme == 'e':
        RGB = np.stack([IRc, F // 2, F], axis=2)
    elif scheme == 'f':
        RGB = np.stack([IRc, F // 2, F // 2], axis=2)
    else:
        RGB = np.stack([IRc, IRc, IRc], axis=2)
    return RGB

# Основні функції для Gradio

def save_image(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(os.path.join(path, filename))

def step1_display_initial_images(vis_image, ir_image):
    Vis = load_image_pil(vis_image)
    IR = load_image_pil(ir_image)
    
    vis_with_hist = show_image_with_histogram(Vis, title='Початкове VIS')
    ir_with_hist = show_image_with_histogram(IR, title='Початкове IR')
    
    # Створення колажу зображення та його гістограми
    collage = Image.new('RGB', (vis_with_hist.width + ir_with_hist.width, max(vis_with_hist.height, ir_with_hist.height)))
    collage.paste(vis_with_hist, (0, 0))
    collage.paste(ir_with_hist, (vis_with_hist.width, 0))
    
    save_image(collage, 'step1/effect/images', 'initial_images_collage.png')
    
    return collage

def step2_contrast_correction(vis_image, ir_image):
    Vis = load_image_pil(vis_image).astype(np.float64)
    IR = load_image_pil(ir_image).astype(np.float64)
    
    Vc = GradientalCorrection(Vis).gradiental_correction()
    IRc = GradientalCorrection(IR).gradiental_correction()
    
    Vc = scale_image(Vc)
    IRc = scale_image(IRc)
    
    Vc_with_hist = show_image_with_histogram(Vc, title='Градаційна Корекція VIS')
    IRc_with_hist = show_image_with_histogram(IRc, title='Градаційна Корекція IR')
    
    # Створення колажу зображення та його гістограми
    collage = Image.new('RGB', (Vc_with_hist.width + IRc_with_hist.width, max(Vc_with_hist.height, IRc_with_hist.height)))
    collage.paste(Vc_with_hist, (0, 0))
    collage.paste(IRc_with_hist, (Vc_with_hist.width, 0))
    
    save_image(collage, 'step2/effect/images', 'contrast_correction_collage.png')
    
    return collage

def step3_fusion_grayscale(vis_image, ir_image):
    # Завантаження та корекція зображень
    Vis = load_image_pil(vis_image).astype(np.float64)
    IR = load_image_pil(ir_image).astype(np.float64)
    
    Vc = GradientalCorrection(Vis).gradiental_correction()
    IRc = GradientalCorrection(IR).gradiental_correction()
    
    F = fusion_methods(Vc, IRc)
    
    # Списки для збереження гістограм та градієнтальних діаграм
    hist_images = []
    gradient_images = []
    
    for method, fused_image in F.items():
        # Створення гістограми для комплексованого зображення
        hist_image = show_image_with_histogram(
            fused_image, 
            title=method, 
            title_left=f'Гістограма {method}'
        )
        hist_images.append(hist_image)

        # Використання GradientalCorrection для корекції градієнтальної величини
        gradient_corrected = GradientalCorrection(fused_image).gradiental_correction()
        
        # Конвертація градієнтальної мапи в зображення
        gradient_image = Image.fromarray(gradient_corrected.astype(np.uint8))
        
        # Створення гістограми для градієнтальної діаграми
        gradient_with_hist = show_image_with_histogram(
            gradient_corrected.astype(np.uint8), 
            title=f'Градаційна корекція {method}',
            title_left=f'Гістограма {method} після градаційної корекції'
        )
        gradient_images.append(gradient_with_hist)
    
    # Визначення розмірів для колажу гістограм (1 колонка)
    hist_width, hist_height = hist_images[0].size
    num_hist = len(hist_images)
    cols_hist = 1
    rows_hist = num_hist  # Кожна гістограма в окремому рядку
    
    # Створення колажу гістограм
    final_hist_collage_width = hist_width * cols_hist
    final_hist_collage_height = hist_height * rows_hist
    final_hist_collage = Image.new('RGB', (final_hist_collage_width, final_hist_collage_height), color='white')
    
    for idx, hist in enumerate(hist_images):
        row = idx // cols_hist
        col = idx % cols_hist
        x_position = col * hist_width
        y_position = row * hist_height
        final_hist_collage.paste(hist, (x_position, y_position))
    
    # Визначення розмірів для колажу градієнтальних діаграм (1 колонка)
    grad_width, grad_height = gradient_images[0].size
    num_grad = len(gradient_images)
    cols_grad = 1
    rows_grad = num_grad  # Кожна градієнтальна діаграма в окремому рядку
    
    # Створення колажу градієнтальних діаграм
    final_grad_collage_width = grad_width * cols_grad
    final_grad_collage_height = grad_height * rows_grad
    final_grad_collage = Image.new('RGB', (final_grad_collage_width, final_grad_collage_height), color='white')
    
    for idx, grad in enumerate(gradient_images):
        row = idx // cols_grad
        col = idx % cols_grad
        x_position = col * grad_width
        y_position = row * grad_height
        final_grad_collage.paste(grad, (x_position, y_position))
    
    # Об'єднання колажів гістограм та градієнтальних діаграм горизонтально
    total_width = final_hist_collage.width + final_grad_collage.width
    total_height = max(final_hist_collage.height, final_grad_collage.height)
    final_combined_collage = Image.new('RGB', (total_width, total_height), color='white')
    final_combined_collage.paste(final_hist_collage, (0, 0))
    final_combined_collage.paste(final_grad_collage, (final_hist_collage.width, 0))
    
    save_image(final_combined_collage, 'step3/effect/images', 'fusion_grayscale_collage.png')
    
    return final_combined_collage

def step4_apply_palettes(vis_image, ir_image):
    Vis = load_image_pil(vis_image).astype(np.float64)
    IR = load_image_pil(ir_image).astype(np.float64)
    
    Vc = GradientalCorrection(Vis).gradiental_correction()
    IRc = GradientalCorrection(IR).gradiental_correction()
    
    F = fusion_methods(Vc, IRc)
    
    # Генерація зображень з палітрами (без Grayscale)
    palette_list = ['Rainbow', 'Fire', 'Thermometer', 'Watermelon', 'Temperature', 'Weather', 'Pumpkin']
    fusion_images = []
    for method, fused_image in F.items():
        for palette in palette_list:
            colored = apply_color_palette(fused_image, palette)
            caption = f"{method} + {palette}"
            fusion_images.append((Image.fromarray(colored), caption))
            save_image(Image.fromarray(colored), f'step4/effect/images/{method}', f'{palette}.png')
    return fusion_images

def step5_synthesize_rgb(vis_image, ir_image):
    Vis = load_image_pil(vis_image).astype(np.float64)
    IR = load_image_pil(ir_image).astype(np.float64)
    
    Vc = GradientalCorrection(Vis).gradiental_correction()
    IRc = GradientalCorrection(IR).gradiental_correction()
    
    F = fusion_methods(Vc, IRc)
    
    schemes = ['a', 'b', 'c', 'd', 'e', 'f']
    synthesized_images = []
    for method, fused_image in F.items():
        for scheme in schemes:
            RGB = synthesize_rgb(IRc, fused_image, scheme)
            RGB = scale_image(RGB)
            caption = f"{method} + Scheme {scheme}"
            synthesized_images.append((Image.fromarray(RGB), caption))
            save_image(Image.fromarray(RGB), f'step5/effect/images/{method}', f'scheme_{scheme}.png')
    return synthesized_images

def step6_informative_images(vis_image, ir_image):
    Vis = load_image_pil(vis_image).astype(np.float64)
    IR = load_image_pil(ir_image).astype(np.float64)
    
    Vc = GradientalCorrection(Vis).gradiental_correction()
    IRc = GradientalCorrection(IR).gradiental_correction()
    
    F = fusion_methods(Vc, IRc)
    
    # Вибір найбільш інформативних методів (Fq, Fa, Fg, Fh, Fch, Fmax, Fmin, Fc)
    informative_methods = ['Fq', 'Fa', 'Fg', 'Fh', 'Fch', 'Fmax', 'Fmin', 'Fc']
    selected_palettes = {
        'Fq': 'Rainbow',
        'Fa': 'Temperature',
        'Fg': 'Fire',
        'Fh': 'Thermometer',
        'Fch': 'Watermelon',
        'Fmax': 'Weather',
        'Fmin': 'Pumpkin',
        'Fc': 'Grayscale'
    }
    
    informative_images = []
    for method in informative_methods:
        if method in F:
            image = F[method]
            if method == 'Fc':
                palette = 'Grayscale'
            else:
                palette = selected_palettes.get(method, 'Grayscale')  # Default to Grayscale if not specified
            colored = apply_color_palette(image, palette)
            caption = f"{method} + {palette}"
            informative_images.append((Image.fromarray(colored), caption))
            save_image(Image.fromarray(colored), f'step6/effect/images/{method}', f'{palette}.png')
    return informative_images

# Створення інтерфейсу Gradio

with gr.Blocks() as demo:
    gr.Markdown("# Комплексування монохромних зображень у кольорові")
    
    with gr.Row():
        vis_input = gr.Image(label="Видиме Зображення (VIS)", type="pil")
        ir_input = gr.Image(label="Інфрачервоне Зображення (IR)", type="pil")
    
    with gr.Row():
        btn_step1 = gr.Button("1. Виведення початкових зображень та їх гістограм")
        btn_step2 = gr.Button("2. Виведення зображень після градаційної корекції та їх гістограм")
        btn_step3 = gr.Button("3. Комплексування зображень у grayscale")
        btn_step4 = gr.Button("4. Накладання палітр на комплексовані зображення")
        btn_step5 = gr.Button("5. Виведення синтезованих RGB зображень за схемами")
        #         btn_step6 = gr.Button("6. Виведення 8 найінформативніших зображень")
    
    with gr.Tab("Крок 1"):
        output1_collage = gr.Image(label="Початкові Зобрження та Гістограми")
    
    with gr.Tab("Крок 2"):
        output2_collage = gr.Image(label="Зображення після Градаційної Корекції та Гістограми")
    
    with gr.Tab("Крок 3"):
        output3_combined_collage = gr.Image(label="Колаж Гістограм та Градієнтальних Діаграм", type="pil")
    
    with gr.Tab("Крок 4"):
        output4_gallery = gr.Gallery(label="Результати Накладання Палітр", columns=7, height="auto")
    
    with gr.Tab("Крок 5"):
        output5_gallery = gr.Gallery(label="Синтезовані RGB Зображення за Схемами", columns=6, height="auto")

    
    # Прив'язка кнопок до функцій
    btn_step1.click(
        step1_display_initial_images, 
        inputs=[vis_input, ir_input], 
        outputs=output1_collage  # Колаж зображень та гістограм
    )
    
    btn_step2.click(
        step2_contrast_correction, 
        inputs=[vis_input, ir_input], 
        outputs=output2_collage  # Колаж коректованих зображень та гістограм
    )
    
    btn_step3.click(
        step3_fusion_grayscale, 
        inputs=[vis_input, ir_input], 
        outputs=output3_combined_collage
    )
    
    btn_step4.click(
        step4_apply_palettes, 
        inputs=[vis_input, ir_input], 
        outputs=output4_gallery
    )
    
    btn_step5.click(
        step5_synthesize_rgb, 
        inputs=[vis_input, ir_input], 
        outputs=output5_gallery
    )


    demo.launch(share=True, debug=True)


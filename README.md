## Комплексування зображень:

Цей проєкт **Image-fusion.py** містить інструменти та методи для виконання градієнтної корекції, комплексування зображень та їх візуалізації з використанням різних технік покращення. 
Робочий процес включає попередню обробку зображень, методи комплексування, застосування кольорових палітр та синтез комплексованих зображень за допомогою RGB моделі.

## Основні можливості
1. **Градієнтна корекція**: Регулювання інтенсивностей вхідних зображень на основі градієнтних порогів.
2. **Методи комплексування**: Збільшення контрастності зображень за допомогою спеціальних алгоритмів.
3. **Застосування палітр**: Застосування палітр до комплексованих зображень у форматі grayscale.
4. **Синтез комплексованих зображень за допомогою RGB моделі**: Створення RGB-зображень із комплексованих зображень за заданими схемами.

---

## Встановлення:

1. Клонування репозиторію:
    ```bash
    git clone https://github.com/irynabystrovaaa/Image-fusion
    cd Image-fusion
    ```

---

## Використання:

### Покроковий робочий процес:

#### 1. **Початкове відображення зображень:**
   Завантаження видимого (VIS) та інфрачервоного (IR) зображень і відображення їх разом з гістограмами.
   ```python
   step1_display_initial_images(vis_image, ir_image)
   ```

#### 2. **Корекція контрасту:**
   Виконання градієнтної корекції для обох зображень (VIS та IR), покращуючи їх контраст.
   ```python
   step2_contrast_correction(vis_image, ir_image)
   ```

#### 3. **Комплексування зображення за допомогою алгоритмів:**
   Комплексуванняя відкорегованих зображення VIS та IR за допомогою алгоритмів комплексування:

   Fmax: Комплексування за максимальним;
   Fmin: Комплексування за мінімальним; 
   Fc: Комплексування за середнім кубічним;
   Fq: Комплексування за середнім квадратичним;
   Fa: Комплексування за середнім арифметичним;
   Fg: Комплексування за середнім геометричним; 
   Fh: Комплексування за середнім гармонійним; 
   Fch: Комплексування за середнім контргармонійним; 
   Ft1: Комплексування за оберненим тангенціальним комплексуванням типу 1; 
   Ft2: Комплексування за зворотнім тангенціальним комплексування типу 2.
   ```python
   step3_grayscale_fusion(vis_image, ir_image)
   ```

#### 4. **Застосування палітр:**
   Перетворення комплексованих зображення у кольоровий вигляд, застосовуючи палітри.
   ```python
   step4_apply_color_palettes(fused_image)
   ```

#### 5. **Синтез комплексованих зображень за допомогою RGB моделі:**
   Створення RGB-зображень із комплексованих зображень за заданими схемами.
   ```python
   step5_rgb_synthesis(grayscale_images)
   ```

---

## Результати:
  Після виконання всіх кроків отримуємо:

  Покращені початкові зображення VIS та IR.
  Комплексовані зображення, що комбінують характеристики обох джерел.
  Комплексовані зображення за допомогою палітр та схем RGB моделі.


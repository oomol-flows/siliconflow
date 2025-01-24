import type { Context } from "@oomol/types/oocana";

type Inputs = {
  token: string;
  model: string;
  prompt: string;
  negative_prompt: string | null;
  image_size: { width: number, height: number } | null;
  batch_size: number | null;
  seed: number | null;
  num_inference_steps: number | null;
  guidance_scale: number | null;
  prompt_enhancement: boolean | null;
}
type Outputs = {
  output: unknown;
}

export default async function (
  params: Inputs,
  context: Context<Inputs, Outputs>
): Promise<Outputs> {
  const { token, model, prompt, negative_prompt, image_size, batch_size, seed, num_inference_steps, guidance_scale, prompt_enhancement } = params;
  const options = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: model,
      prompt: prompt,
      negative_prompt: negative_prompt !== null ? negative_prompt : "<string>",
      image_size: image_size !== null ? `${image_size.width}x${image_size.height}` : "1024x1024",
      batch_size: batch_size !== null ? batch_size : 1,
      seed: seed !== null ? seed : 4999999999,
      num_inference_steps: num_inference_steps !== null ? num_inference_steps : 20,
      guidance_scale: guidance_scale !== null ? guidance_scale : 7.5,
      prompt_enhancement: prompt_enhancement !== null ? prompt_enhancement : false,
    }),
  };

  try {
    const response = await fetch('https://api.siliconflow.cn/v1/images/generations', options);
    const data = await response.json();
    console.log(data);
    const images = data.images.map((image: { url: string }) => image.url);
    context.preview({
      type: "image",
      data: images
    })
  } catch (err) {
    console.error(err);
  }

  return { output: null };
};

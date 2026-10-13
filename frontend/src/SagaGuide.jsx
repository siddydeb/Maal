const ORDER = [
  ["Who", "1girl, solo, mature female, adult woman"],
  ["Body", "curvy, slim, tall, muscular female, wide hips"],
  ["Face", "freckles, mole under eye, red lips, eyeshadow, glasses"],
  ["Hair", "long hair, short hair, ponytail, hair bun, brown hair, blonde hair"],
  ["Eyes", "blue eyes, green eyes, closed eyes, looking at viewer"],
  ["Expression", "smile, grin, open mouth, blush, angry, crying, surprised"],
  ["Clothes", "red dress, white shirt, jeans, apron, bikini, office suit, stockings"],
  ["Pose / action", "standing, sitting, hand on hip, arms crossed, waving, holding cup"],
  ["Place", "kitchen, bedroom, classroom, beach, park, city street, indoors, outdoors"],
  ["Light / time", "morning, sunset, night, sunlight, lamp light, dim lighting"],
];

const EXAMPLES = [
  ["a tired nurse in her 40s sitting on a hospital bench at night drinking coffee",
    "1girl, nurse, mature female, sitting, hospital bench, night, drinking coffee"],
  ["make her eyes blue and give her darker lipstick, a bit more smile", "blue eyes, dark lipstick, smiling"],
  ["she is angry and crying, mascara running", "angry, crying, running mascara"],
  ["put her in a red bikini at the beach, sunset", "red bikini, beach, sunset"],
  ["face", "face focus, close-up, portrait"],
];

const AVOID = [
  ["Saying what you don't want", "\"no hat\" can add a hat, because the model sees the word hat. Just leave it out."],
  ["Long stories or reasons", "\"because she just got home from work\" means nothing to it. Describe only what is visible."],
  ["Names it doesn't know", "Your own character names (e.g. NewCharacter, replacing Jenny) are just noise. Describe her look instead, or pick a canon character from the list."],
  ["Exact counts and text", "\"exactly three rings\", signs, logos and written words come out wrong."],
  ["Quality words", "masterpiece, best quality and the style trigger are added for you. Adding them again does nothing."],
];

export default function SagaGuide() {
  return (
    <section className="panel view-panel saga-guide">
      <h2 className="section-title">How the model reads your prompt</h2>
      <p>
        The Saga model was trained on anime pictures labelled with short <b>tags</b> (the Danbooru style):
        <code>1girl, long hair, red dress, smile, kitchen</code>. It understands simple plain English too,
        but tags are what it knows best. With <b>"Turn my plain English into model tags"</b> switched on
        (Studio tab, under the description), your words are converted to tags for you. Every result shows
        what the model actually read under <b>"Model read"</b> when you open it, so you can learn which
        words it picked up.
      </p>
      <p>
        If you already write tags (four or more short comma-separated pieces), they are sent as they are.
      </p>

      <h3>Order that works best</h3>
      <p>Most important first. Separate every detail with a comma.</p>
      <table className="saga-guide-table">
        <tbody>
          {ORDER.map(([what, tags]) => (
            <tr key={what}>
              <th>{what}</th>
              <td>
                <code>{tags}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Plain English → what the model gets</h3>
      <table className="saga-guide-table">
        <tbody>
          {EXAMPLES.map(([plain, tags]) => (
            <tr key={plain}>
              <th>{plain}</th>
              <td>
                <code>{tags}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>What doesn't work</h3>
      <ul>
        {AVOID.map(([what, why]) => (
          <li key={what}>
            <b>{what}.</b> {why}
          </li>
        ))}
      </ul>

      <h3>Ages</h3>
      <p>
        Everyone is an adult. Ages like "35 year old" are fine. Useful tags: <code>adult woman</code>,{" "}
        <code>mature female</code> (30s and up), <code>milf</code>, <code>old woman</code>. Anything that reads
        as under 18 is refused.
      </p>

      <h3>Editing an image (photo or part)</h3>
      <ul>
        <li>
          <b>Description box</b>: what is in the image (helps the model read it correctly).
        </li>
        <li>
          <b>Changes box</b>: what to change, e.g. <code>green eyes, freckles, darker lipstick</code>. Keep it to
          things you can see.
        </li>
        <li>
          <b>Strength</b>: Light keeps shape and features closest; Medium redraws more and is needed for
          changes (Light is raised to Medium automatically when you ask for changes); Strong for big changes
          like a new hairstyle or outfit.
        </li>
        <li>
          <b>Part type</b> (Part layout): pick Face, Hand, Body or Hair so a lone cut-out is not mistaken for
          something else. Part mode zooms into the visible pixels, redraws them in full detail, and puts them
          back at the same spot and size on your canvas.
        </li>
      </ul>

      <h3>Canon characters</h3>
      <p>
        Pick a character from the list instead of describing her. Her look is added for you, and her usual
        outfit too if that box is ticked. Then only describe what she is doing and where.
      </p>
    </section>
  );
}

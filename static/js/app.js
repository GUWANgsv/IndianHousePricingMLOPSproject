document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('predict-form')
  const result = document.getElementById('result')

  form.addEventListener('submit', async function (e) {
    e.preventDefault()
    result.textContent = 'Predicting from form inputs...'

    const formData = new FormData(form)
    const payload = Object.fromEntries(formData.entries())

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await resp.json()
      if (resp.ok) {
        result.innerHTML = '<strong>Estimated price:</strong> ' + Number(data.prediction).toLocaleString(undefined, {
          maximumFractionDigits: 2,
        })
      } else {
        result.textContent = 'Error: ' + (data.error || 'Unknown error')
      }
    } catch (err) {
      result.textContent = 'Request failed: ' + err
    }
  })
})

const handleSendMessage = async (message) => {
  setIsLoading(true);
  try {
    // Tu lógica para enviar el mensaje
    await sendMessage(message);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    setIsLoading(false);
  }
}; 
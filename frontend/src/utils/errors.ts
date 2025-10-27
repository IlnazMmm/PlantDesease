import axios from "axios";

export function getErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const responseData = error.response?.data;
    if (typeof responseData === "string") {
      return responseData;
    }
    if (responseData && typeof responseData === "object" && "message" in responseData) {
      const { message } = responseData as { message?: unknown };
      if (typeof message === "string") {
        return message;
      }
    }
    if (typeof error.message === "string" && error.message.trim().length > 0) {
      return error.message;
    }
  }

  return fallback;
}
